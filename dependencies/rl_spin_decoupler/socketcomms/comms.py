"""

COMMUNICATIONS BASED ON SOCKETS

(c) Juan-Antonio Fernández-Madrigal, 2025
https://babel.isa.uma.es/jafma

"""

from enum import Enum
import ipaddress
import struct
import time
import datetime
import socket,pickle,select
from typing import Dict,List,Tuple


# -----------------------------------------------------------------------------
#
#	Base class: BaseCommPoint
#
# -----------------------------------------------------------------------------

class BaseCommPoint:
	"""
	Communication point.
	"""
	_LEN_STRUCT = struct.Struct("!Q")  # 8-byte unsigned length, network byte order

	class Kind(Enum):
		"""
		Kinds of points
		"""
		SERVER = 0	
		CLIENT = 1	


	@classmethod
	def get_ip(cls):
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.settimeout(0)
		try:
			# doesn't even have to be reachable
			s.connect(('10.254.254.254', 1))
			IP = s.getsockname()[0]
		except Exception:
			IP = '127.0.0.1'
		finally:
			s.close()
		return IP		
		
	
	def __init__(self, kind: Kind , datachunkmaxsize: int = 4096, port: int = 49054, ipv4: str = "127.0.0.1"):
		"""
		Constructor. The point is set at the given port and machine IPv4.
		"""
		if not isinstance(kind,BaseCommPoint.Kind):
			raise(TypeError("Expected a Kind argument, got {}".format(type(kind))))
		if (not isinstance(datachunkmaxsize, int)) or not (0 < datachunkmaxsize):
			raise(ValueError("The max. size of data chunks {} is invalid".format(datachunkmaxsize)))			
		if (not isinstance(port, int)) or not (20000 <= port <= 49151):
			raise(ValueError("Port {} is invalid; it should be an integer between 20000 and 49151".format(port)))			
		try:
			ipaddress.IPv4Address(ipv4)
		except ipaddress.AddressValueError:
			raise(ValueError("IP address {} is invalid".format(ipv4)))
			
		self._kind = kind
		self._datachunkmaxsize = datachunkmaxsize
		self._port = port
		self._ipv4 = ipv4
		self._begun = False # to be set in derived classes
		self._debug = False
		
	def __copy__(self):
		"""
		Prevent to make copies or deepcopies.
		"""
		raise NotImplementedError("Cannot do copies of CommPoint")
		
	def _printInfo(self,info:str):
		now = datetime.datetime.now()
		print("CommPoint[" + str(now) + "]: " + info,flush=True)
		
	def setDebug(self,st:bool = True):
		"""
		Enable or disable debug messages.
		"""
		self._debug = st

	def _recv_exact(self, nbytes: int) -> bytes:
		"""Receive exactly nbytes from the socket.

		Args:
			nbytes (int): Number of bytes to read.

		Returns:
			bytes: The received bytes.

		Raises:
			RuntimeError: If the connection is closed before receiving enough data.
		"""
		chunks = []
		remaining = nbytes
		while remaining > 0:
			chunk = self._sock.recv(min(self._datachunkmaxsize, remaining))
			if chunk == b"":
				raise RuntimeError("Connection closed while receiving")
			chunks.append(chunk)
			remaining -= len(chunk)
		return b"".join(chunks)
					
	def sendData(self, data: Dict) -> str:
		"""
		Send that data properly to the other side.
		Return non-empty string with any error in the connection.
		"""
		if not self._begun:
			raise RuntimeError("Cannot send data in not-begun commpoint")
		
		try:
			payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

            # Prefix with payload length to delimit messages over TCP stream.
			header = BaseCommPoint._LEN_STRUCT.pack(len(payload))
			
			if self._debug:
				self._printInfo(f"Sending framed msg: header={len(header)} bytes, payload={len(payload)} bytes...")

            # sendall guarantees full transmission.
			self._sock.sendall(header)
			self._sock.sendall(payload)
			if self._debug:
				self._printInfo("\tSent ok.")
			return ""
		except Exception as e:
			return str(e)
		
	def readData(self, timeout: float = 2.0) -> Tuple[str, Dict]:
		"""
		Read one message (blocking if timeout > 0.0) from the other side.
		Return non-empty string if any error in the connection.
		"""
		if not self._begun:
			raise RuntimeError("Cannot send data in not-begun commpoint")

		if timeout <= 0.0:
			timeout = None
		self._sock.settimeout(timeout)

		try:
			if self._debug:
				self._printInfo("Receiving...")

			# Read framed header (8 bytes) then payload.
			header = self._recv_exact(BaseCommPoint._LEN_STRUCT.size)
			(msg_len,) = BaseCommPoint._LEN_STRUCT.unpack(header)

			payload = self._recv_exact(msg_len)
			result = pickle.loads(payload)

			if self._debug:
				self._printInfo(f"\tReceived framed msg: payload={msg_len} bytes.")
			res = ""
		except Exception as e:
			result = None
			res = str(e)

		self._sock.settimeout(None)
		return res, result
		
	def checkDataToRead(self):
		"""
		Check whether the socket has data to read and return True in that case.
		This is a non-blocking test.
		"""
		if not self._begun:
			raise RuntimeError("Cannot send data in not-begun commpoint")
		if self._debug:
			self._printInfo("Peeking...")
		ready_to_read, _, _ = select.select([self._sock], # sockets to check for reading
											[], [], # writes and exceptions to check 
											0) # non-blocking
		if ready_to_read:
			return True
		return False
			


# -----------------------------------------------------------------------------
#
#	Class: ServerCommPoint
#
# -----------------------------------------------------------------------------

class ServerCommPoint(BaseCommPoint):
	
	def __init__(self, po: int):
		"""
		Constructor. Server listening at that port.
		"""
		self._servip = BaseCommPoint.get_ip()
		super().__init__(kind = BaseCommPoint.Kind.SERVER, port = po, ipv4 = self._servip)
		finish = False
		tries = 0
		while not finish:
			try:
				self._basesock = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # 1st arg: ip4, 2nd: TCP
				self._basesock.bind((self._ipv4,self._port)) # does not block
				finish = True
			except socket.error as e:
				if e.errno == socket.errno.EADDRINUSE:
					tries += 1
					if tries > 10:
						print("Too many tries. Aborting")
						raise
					print(f"Port {self._servip}:{po} already in use. Retrying in 13 secs ({tries})...")
					time.sleep(13) # wait to retry
				else:
					print(f"Socket error: {e}")
					raise
		self._basesock.listen(1) # does not block
		print("---> Server comm point listening")
		
	def __str__(self) -> str:
		return "Server listening at {}:{}, began: {}".format(self._servip,self._port,self._begun)
		
	def begin(self,timeoutaccept: float) -> str:
		"""
		Start the work for the server.
		TIMEOUTACCEPT in seconds.
		"""
		if timeoutaccept <= 0.0:
			raise ValueError("Timeoutaccept must be > 0.0")
		if not self._begun:
			self.end()
		self._basesock.settimeout(timeoutaccept) # after this, we assume the other side has shut down
		try:
			self._sock, _ = self._basesock.accept() # wait for calling us
			self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # disable Nagle's algorithm to reduce latency spikes
			self._begun = True
			self._basesock.settimeout(None) # to deactivate timeout in other operations
			return ""
		except socket.timeout:
			self._basesock.settimeout(None) # to deactivate timeout in other operations
			return "timeout"
		except Exception as e:
			return str(e)
			
	def end(self) -> str: 
		"""
		Ends the communications for the current work.
		"""
		if self._begun:
			try:
				self._sock.close()			
				self._begun = False
				return ""
			except Exception as e:
				return str(e)
		return ""
				
		
		
# -----------------------------------------------------------------------------
#
#	Class: ClientCommPoint
#
# -----------------------------------------------------------------------------

class ClientCommPoint(BaseCommPoint):
	
	def __init__(self, ip: str, po: int):
		"""
		Constructor. Client to connect to that ip:port.
		"""
		self._myip = BaseCommPoint.get_ip()
		super().__init__(kind = BaseCommPoint.Kind.CLIENT, ipv4 = ip, port = po)

	def __str__(self) -> str:
		return "Client at {} to connect to {}:{}, began: {}".format(self._myip,self._ipv4,self._port,self._begun)
		
	def begin(self) -> str:
		"""
		Start the work for the client.
		"""
		if not self._begun:
			self.end()
		try:
			self._sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # 1st arg: ip4, 2nd: TCP
			self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # disable Nagle's algorithm to reduce latency spikes
			self._sock.connect((self._ipv4,self._port)) # if bind-listen has been done on the other side but accept has not, ends immediately even when the server is not accpeting at the time (connection is kept pending), and data can be sent; if bind-listen has not been done on the other side, an error is raised
			self._begun = True
			return ""
		except Exception as e:
			return str(e)
		
	def end(self) -> str: 
		"""
		Ends the communications for the current work.
		"""
		if self._begun:
			try:
				self._sock.close()			
				self._begun = False
				return ""
			except Exception as e:
				return str(e)
		return ""

	
	
if __name__=='__main__':

	user_input = input("IP to connect to (empty if server): ")
	if len(user_input) == 0:
		comm = ServerCommPoint(49054)
		print("[{}] prepared to begin".format(str(comm)))
		if not comm.begin(60.0):
			raise RuntimeError("No one has connected to this server before timeout")
		print("[{}] connected".format(str(comm)))
		ind = 0
		while True:
			data = comm.readData(30.0)
			if data[0]:
				print("\t#{}. Received data {}".format(ind,data[1]))
			else:
				raise RuntimeError("\t#{}. Some error receiving data".format(ind))
			comm.sendData(data[1])
			print("\t\tSent response")
			ind += 1
	else:
		port = input("Port to connect to: ")
		comm = ClientCommPoint(ip = user_input,po = int(port))
		print("[{}] prepared to connect".format(str(comm)))
		comm.begin()
		print("[{}] connected".format(str(comm)))
		ind = 0
		while True:
			comm.sendData({"d":54.54,"i":ind})
			print("\t\tSent data")
			data = comm.readData(30.0)
			if data[0]:
				print("\t#{}. Received response {}".format(ind,data[1]))
			else:
				raise RuntimeError("\t#{}. Some error receiving response".format(ind))
			time.sleep(10)
			ind += 1
		
