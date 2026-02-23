"""
SYSTEM FOR DECOUPLING RL SPIN LOOP FROM AGENT SPIN LOOP.

v1.2.0

(c) Juan-Antonio Fernández-Madrigal
Uncore Team, 2025
"""

from enum import Enum
from socketcomms.comms import ClientCommPoint,ServerCommPoint


#-------------------------------------------------------------------------------
#
#	Base Class: RLSide
#
#-------------------------------------------------------------------------------

class RLSide:
	"""
	Your RL class must create an instance of this in order to communicate
	with the agent that actually produces observations and executes actions.
	"""
	
	def __init__(self, port: int, verbose: bool = False):
		"""
		In PORT, the number of the port to use for comms., e.g., 49054.
		"""
		
		self._verbose = verbose
		self._rlcomm = ServerCommPoint(port) # socket not connected yet
											 # if socket in use, repeatedly wait
											 # until free
		if self._verbose:
			print("RL decoupler enabled. Waiting for agent connection...")
		res = self._rlcomm.begin(timeoutaccept = 60.0) # blocks for agent
		if len(res) > 0:
			raise RuntimeError("No agent connection: " + res)
		if self._verbose:
			print("Agent connected to this RL")
			
			
	def __del__(self):
	
		res = self._rlcomm.end()
		if len(res) > 0:
			print("Error closing communications with the agent: " + res)
		if self._verbose:
			print("Communications closed in the RL side.")

		
	def resetGetObs(self, timeout: float = 10.0):
		"""
		Call this method at the start of your RL reset() to request from the 
		agent the first observation after a reset, as a dictionary. The caller
		gets blocked until the agent sends the observation.
		TIMEOUT is the timeout in seconds used for communication operations that
		admit a timeout. If it is <0.0, no timeout is checked.
		It raises RuntimeError() if any error in communications.
		Return both the observation (a dictionary) and the agent time when that
		observation was gathered (a float).
		"""
		
		res = self._rlcomm.sendData(dict({"stepkind": "reset"}))
		if len(res) > 0:
			raise RuntimeError("Error sending what to do to the agent. " + res)	
			
		res,obs = self._rlcomm.readData(timeout)
		if len(res) > 0:
			raise RuntimeError("Error reading after-reset observation from "
							   "the agent. " + res)
					
		return obs["obs"], obs["ato"], obs["info"] # return observation + ato + extra info

	
	def stepSendActGetObs(self, action,timeout:float = 10.0):
		"""
		Call this method at the start of your RL step() to send the action
		to the agent and then get the resulting observation, both as 
		dictionaries. The caller gets blocked until the agent executes the 
		action and sends back the observation. Along with the observation, it
		returns the total duration of the action previous to this one, the 
		reward calculated by the agent for the current action and the agent
		time when it got the observation.
		TIMEOUT is the timeout in seconds used for communication operations that
		admit a timeout. If it is <0.0, no timeout is checked.
		It raises RuntimeError() if any error in communications.
		""" 
		
		# send a STEP indicator to the agent interface, that should use
		# readWhatToDo() to get the indicator
		res = self._rlcomm.sendData(dict({"stepkind": "step",
										  "action": action}))
		if len(res) > 0:
			raise RuntimeError("Error sending step action: " + res)

		res,lat = self._rlcomm.readData(timeout) # blocks
		if len(res) > 0:
			raise RuntimeError("Error receiving last action duration: " + res)
			
		res,obsrewato = self._rlcomm.readData(timeout) # blocks
		if len(res) > 0:
			raise RuntimeError("Error receiving step observation: " + res)

		return lat["lat_sim"], lat["lat_wall"], obsrewato["obs"], obsrewato["rew"], obsrewato["ato"]

				
	def stepExpFinished(self, timeout:float = 10.0):
		""" 
		Call this method at the end of your RL step() ONLY IF the learning
		has finished completely after that step.
		TIMEOUT is the timeout in seconds used for communication operations that
		admit a timeout. If it is <0.0, no timeout is checked.
		"""
		
		self._rlcomm.sendData(dict({"stepkind": "finish"}))

	
#-------------------------------------------------------------------------------
#
#	Base Class: AgentSide
#
#-------------------------------------------------------------------------------

class AgentSide:
	"""
	Your agent interface (e.g., with a robot or a simulation), in charge of
	getting observations and executing actions, must contain an instance of
	this class to communicate with the RL spin loop, which provides the actions.
	"""
	
	class WhatToDo(Enum):
		"""
		Things that RL is intending for the agent interface to do.
		"""
		
		REC_ACTION_SEND_OBS = 0	# receive action from RL, executes it and sends 
								# back resulting observation and other stuff
		RESET_SEND_OBS = 1		# reset episode and send observation back to RL
		FINISH = 2				# finish experiment (and comms)
	
	
	def __init__(self, ipbaselinespart:str, 
				 portbaselinespart:int, 
				 verbose:bool = False):
		"""
		IPBASELINESPART is the IPv4 of the baselines part of the system, e.g.,
		"BaseCommPoint.get_ip()".
		PORTBASELINESPART is the port, e.g., 49054.
		"""
		
		self._verbose = verbose
		self._rlcomm = ClientCommPoint(ipbaselinespart,portbaselinespart)
		
		if self._verbose:
			print("Agent decoupler enabled.")
		
		res = self._rlcomm.begin()
		if len(res) > 0:
			raise RuntimeError("Error starting connection with RL. " + res)
		
		if self._verbose:
			print("Agent decoupler connected to RL decoupler")
		
					
	def __del__(self):
	
		res = self._rlcomm.end()
		if len(res) > 0:
			raise RuntimeError("Error stopping connection with RL: " + res)
		if self._verbose:
			print("Connection with RL finished.")

 	
	def readWhatToDo(self, timeout:float = 10.0): 	
		""" 
		Call this method at each iteration of the agent spin loop if you need
		to receive new commands from the RL side. 
		It returns a tuple with an indicator plus possibly some data received
		(or None if none). If nothing to do is pending in the communications 
		channel, return None, thus this method is not blocking except when there
		are data in the channel.
		Depending on the indicator, you must:
			REC_ACTION_SEND_OBS : take the action from the second element of the
								  tuple, start its execution, return the actual
								  duration of the action executed before that
								  (see stepSendLastActDur()), execute the new
								  action during some time, and then send back an 
								  observation read after that time (see 
								  stepSendObs()).
			RESET_SEND_OBS:	reset the episode for the agent, read an observation
							turn it into a dictionary and call the 
							resetSendObs() method with that.
			FINISH:	nothing besides the needed final arrangements of the agent
					to finish the experiment (the comms are closed automatically
					in this case).
		TIMEOUT is the maximum time to take for the reading of the channel.
		This method can raise RuntimeError if any error occurs in comms.
		"""
		
		if not self._rlcomm.checkDataToRead():
			return None
		
		# read last (pending) step()/reset() msg and then proceed accordingly
		res,ind = self._rlcomm.readData(timeout) 
		# read a dict: { 'stepkind' : 'reset', 'step' or 'finish' ,
		#			      'action' : <action> if any}
		if len(res) > 0:
			raise RuntimeError("Error receiving what-to-do from RL: " + res)
				
		if ind["stepkind"] == "step":
			return (AgentSide.WhatToDo.REC_ACTION_SEND_OBS, ind["action"])
		elif ind["stepkind"] == "reset":
			return (AgentSide.WhatToDo.RESET_SEND_OBS, None)
		elif ind["stepkind"] == "finish":
			return (AgentSide.WhatToDo.FINISH, None)
		else:
			raise(ValueError("Unknown what-to-do indicator [" + 
							 ind["stepkind"] + "]"))

	def stepSendLastActDur(self, lat_sim:float, lat_wall:float):
		"""
		Call this method after receiving a REC_ACTION_SEND_OBS and starting the
		action, being LAT the actual time during which the action previous to 
		that one was executed (before being substituted by the one in 
		REC_ACTION_SEND_OBS).
		This method can raise RuntimeError if any error occurs in comms.
		"""

		res = self._rlcomm.sendData(dict({"lat_sim": lat_sim, "lat_wall":lat_wall}))
		if len(res) > 0:
			raise RuntimeError("Error sending lat to RL. " + res)	


	def stepSendObs(self, obs, agenttime:float = 0.0, rew:float = 0.0):		
		"""
		Call this method if readWhatToDo() returned REC_ACTION_SEND_OBS, after
		executing the action, with the observation (a dictionary) to be sent 
		back to the RL and reward calculated for that action, if any (usually,
		reward is calculated at the RL side, but in some situations it could be
		interesting to calculate it at the agent side).
		Agenttime is the time when the agent got the observation.
		This method can raise RuntimeError if any error occurs in comms.
		"""
		
		res = self._rlcomm.sendData(dict({"obs":obs,"rew":rew,"ato":agenttime}))
		if len(res) > 0:
			raise RuntimeError("Error sending observation/reward to RL. " + res)	

					
	def resetSendObs(self,obs,agenttime = 0.0, extra_info = {}):
		"""
		Call this method if readWhatToDo() returned RESET_SEND_OBS to send back
		the first observation (OBS, a dictionary) got after an episode reset,
		along with the time (of the agent) when that observation was gathered.
		This method can raise RuntimeError if any error occurs in comms.
		"""

		res = self._rlcomm.sendData({"obs":obs,"ato":agenttime, "info": extra_info})
		if len(res) > 0:
			raise RuntimeError("Error sending observation to RL. " + res)	
		


#-------------------------------------------------------------------------------
#
#	Base Class: RLSideQuery
#
#-------------------------------------------------------------------------------			 	


class RLSideQuery:
	"""
	Just answers queries from the agent.

	"""
	def __init__(self, port: int, verbose: bool = False):
		self._verbose = verbose
		self._srv = ServerCommPoint(port)
		if self._verbose:
			print(f"[Query Server] Waiting for agent query connection on port {port}...")
		res = self._srv.begin(timeoutaccept=60.0)
		if len(res) > 0:
			raise RuntimeError("No agent connection for query channel: " + res)
		if self._verbose:
			print("[Query Server] Agent connected for queries.")
	
	def __del__(self):
		res = self._srv.end()
		if len(res) > 0:
			print("Error closing query channel (RL side): " + res)
		if self._verbose:
			print("Communications closed in the RL side.")

	def reconnect(self, timeoutaccept: float = 60.0):
		"""
		Closes the current connection and waits for a new agent connection.
		"""
		res = self._srv.end()
		if len(res) > 0:
			raise RuntimeError("Error closing query channel (RL side): " + res)
		if self._verbose:
			print("[Query Server] Waiting for agent query reconnection...")
		res = self._srv.begin(timeoutaccept=timeoutaccept)
		if len(res) > 0:
			raise RuntimeError("No agent reconnection for query channel: " + res)
		if self._verbose:
			print("[Query Server] Agent reconnected for queries.")


	def receive_query(self, timeout: float = -1.0):
		"""
		Blocks until receiving a new query or until timeout ends (if >=0).
		Returns the received dict (e.g.: {"stepkind":"query","obs":{...}}).
		"""
		res, msg = self._srv.readData(timeout)
		if len(res) > 0:
			raise RuntimeError("Error reading query from agent: " + res)
		return msg


	def send_action(self, action_dict):
		"""
		Sends the predicted action to the agent as a dictionary.
		"""
		res = self._srv.sendData({"action": action_dict})
		if len(res) > 0:
			raise RuntimeError("Error sending action to agent (query channel): " + res)


	def wait_for_query(self, timeout: float = -1.0) -> bool:
		"""Block until a 'query' flag arrives (or timeout expires).

		Args:
			timeout (float): Maximum time in seconds to wait for data.
				A negative value means "block indefinitely".

		Returns:
			bool: True if a valid query flag was received, False if no data
			were available within the given timeout (when timeout >= 0).

		Raises:
			RuntimeError: On communication errors.
			ValueError: On unexpected message format or stepkind.
		"""
		# Non-blocking behavior when timeout >= 0 and no data pending:
		if timeout >= 0.0 and not self._srv.checkDataToRead():
			return False

		res, msg = self._srv.readData(timeout)
		if len(res) > 0:
			raise RuntimeError("Error reading query from agent: " + res)

		if not isinstance(msg, dict):
			raise ValueError(f"[Query Server] Unexpected message type: {type(msg)}")

		stepkind = msg.get("stepkind", None)
		if stepkind != "query":
			raise ValueError(
				f"[Query Server] Unexpected stepkind while waiting for query: {stepkind}"
			)

		if self._verbose:
			print("[Query Server] Query flag received.")

		return True


#-------------------------------------------------------------------------------
#
#	Base Class: AgentSideQuery
#
#-------------------------------------------------------------------------------		


class AgentSideQuery:
	"""
	Client for sending virtual observations to the RL side and received their 
	corresponding actions.
	"""

	def __init__(self, ip_rl: str, port_rl: int, verbose: bool = False):
		self._verbose = verbose
		self._comm = ClientCommPoint(ip_rl, port_rl)
		res = self._comm.begin()
		if len(res) > 0:
			raise RuntimeError("[Query client] Error connecting query channel to RL: " + res)
		if self._verbose:
			print("[Query client] Agent connected to RL query channel.")
	
	def __del__(self):
		res = self._comm.end()
		if len(res) > 0:
			print("[Query client] Error closing query channel (agent side): " + res)
		if self._verbose:
			print("[Query client] Communications closed in the Agent side.")

	def query_action(self, obs_dict, timeout: float = 10.0):
		"""
		Sends the observations and receives an action in response.
		"""
		res = self._comm.sendData({"stepkind": "query", "obs": obs_dict})
		if len(res) > 0:
			raise RuntimeError("[Query client] Error sending query obs to RL: " + res)

		res, msg = self._comm.readData(timeout)
		if len(res) > 0:
			raise RuntimeError("[Query client] Error receiving query action from RL: " + res)

		return msg["action"]
	

	def send_query(self):
		"""Send a 'query' flag to the RL side.

		The message is a small dictionary with the single field::

			{"stepkind": "query"}

		The RL side will interpret this as a request to perform exactly one
		evaluation/test step using its own environment and observation.
		"""
		res = self._comm.sendData({"stepkind": "query"})
		if len(res) > 0:
			raise RuntimeError(
				"[Query client] Error sending query flag to RL: " + res
			)

		if self._verbose:
			print("[Query client] Query flag sent to RL.")
