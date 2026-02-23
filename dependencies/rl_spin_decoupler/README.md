# RL Spin Decoupler


This is a simple Python module that allows to sync an **RL algorithm** (e.g., from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)) to an **agent**, physical or simulated, that is able to get observations and execute actions.

Usually, this would not be necessary since in many cases you can implement in the same environment class all what is needed for both RL and the agent, but there are scenarios where the agent needs to execute some *spin loop*, in its own thread, while RL executes its own. Both loops may be quite difficult to put in sync. **RL Spin Decoupler** is intended for those cases.

Its use is pretty simple: you will have a Python program running RL and another one -a different process- running the agent. The former will use the `RLSide` class of this module, and the latter the `AgentSide` class in order to communicate to each other (communications are implemented with [sockets](https://docs.python.org/3/library/socket.html)). You can find further explanations in the code about when and how to call the methods of these classes in order to sync both processes. 

In addition, two files called 'skeleton_...' contain incomplete implementations of a case of use of the decoupler.

Besides facilitating the link between different processes that carry out RL and agent simulation/control, this decoupler may be useful as well when **different timings are involved and matter** (for instance, if the agent must execute some action at a given time, while the RL algorithm has no notion of that).

Although **RL Spin Decoupler** has been implemented and tested with Stable Baselines3 in mind, it is quite general and could be used with other RL libraries: it assumes that there is a `step()` method at each RL step and a `reset()` method when an episode starts. The parameters and results of those methods are the same that Stable Baselines3 uses.
