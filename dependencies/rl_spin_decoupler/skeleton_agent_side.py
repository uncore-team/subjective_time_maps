""" 
AGENT PART OF THE RL-AGENT DECOUPLING EXAMPLE
Start this after the other.
"""

# imports
from enum import Enum
from spindecoupler import AgentSide



class Agent:
	"""The agent"""

	class StepState(Enum):
		"""
		States of the agent when enters its step() method
		"""
		READYFORRLCOMMAND = 0	# Ready for a new RL command
		EXECUTINGLASTACTION = 1	# Executing the last action
		AFTERRESET = 2	# After a previous (immediate) reset


	def __init__(self, debug = False) -> None:
	
		self._control_timestep = ... # timestep of the step() cycle in the agent
		self._rltimestep = ... # timestep of a RL step (not of an Agent step; the timestep of the Agent step is self._control_timestep)
		if self._rltimestep <= self._control_timestep:
			raise(ValueError("RL timestep must be > control timestep"))

		self._stepstate = Agent.StepState.READYFORRLCOMMAND
		self._lastaction = None
		self._lastactiont0 = 0.0
		self._starttimecurepisode = 0.0
			
		self._commstoRL = AgentSide(BaseCommPoint.get_ip(),49054,verbose = debug) # wait til connecting to RL
	
		self._debug = debug
		
		print("Agent created.")



	def step(self, timestep) -> np.ndarray:
		"""A step for the agent: timestep includes the observation).
		It also manages episodic resets."""

		if self._debug:
			print("Step")

		act = self._lastaction	# by default, continue executing the same last action
		curtime = ... # get aget current time

		if self._stepstate == Agent.StepState.EXECUTINGLASTACTION: 
			# --- not waiting new commands from RL, just executing last action
		
			if (curtime - self._lastactiont0 >= self._rltimestep): 
				# last action is finished by now
			
				observation = ... # gather observation
				self._commstoRL.stepSendObs(observation,curtime) 
				self._stepstate = Agent.StepState.READYFORRLCOMMAND

		elif self._stepstate == Agent.StepState.READYFORRLCOMMAND: 
			# --- waiting for new RL step() or reset() command from RL
		 
			# read the last (pending) step()/reset() indicator 
			whattodo = self._commstoRL.readWhatToDo()
			if self._debug:
				print("\tNew command: {}".format(whattodo))
			if whattodo is not None: # otherwise, no command available yet
					
				if whattodo[0] == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:

					actrec = whattodo[1]

					lat = curtime - self._lastactiont0
					self._lastactiont0 = curtime
					self._commstoRL.stepSendLastActDur(lat)
					self._stepstate = Agent.StepState.EXECUTINGLASTACTION 
					# from now on, we are executing that action

				elif whattodo[0] == AgentSide.WhatToDo.RESET_SEND_OBS:

					... # do reset the agent scenario / episode

					act = ... # null action
					self._starttimecurepisode = curtime
					self._stepstate = Agent.StepState.AFTERRESET 
					# prepare to send an observation right after this

				elif whattodo[0] == AgentSide.WhatToDo.FINISH:
				
					raise RuntimeError("Experiment finished")
					
				else:
					raise(ValueError("Unknown indicator data"))

		elif self._stepstate == Agent.StepState.AFTERRESET: 
			# --- must send the pending observation after the last reset
		
			observation = ... # gather observation
			self._commstoRL.resetSendObs(observation,curtime)
			self._stepstate = Agent.StepState.READYFORRLCOMMAND
			if self._debug:
				print("\tObservation sent after reset: {}".format(observation))
				
				
		if self._debug:
			print("Step panda -- end")
		self._lastaction = act
		return act	 # to be executed now by Panda



if __name__ == '__main__':

	# initialize agent

	"""
	main loop
	"""

	agent = Agent(env,False)

	agent.spinloop()...


