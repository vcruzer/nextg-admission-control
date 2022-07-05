# Admission Control in NextG Scenario with Reinforcement Learning

The goal is to have scenario/environment where requests will arrive needing resources from the Provider (INP) and it needs to decide which ones to accept or reject. We will be using Reinforcement Learning to maximize the INP profit. The pricing model is dynamic and changes depending on a set of different factors such as Supply,Demand,Priority, Resource Usage and job duration.

## About Folders
- *./dataset* contains datasets generated through "generate_dset.py" source for deterministic experiments
- *./src* contains the code and libraries for running a fcfs and the RL versions
- *./params* contains the csv parameters generated through random_search.py performed via RayTune


## Sources
- *./RL_torch.py* is the code that calls and manages the environment and DDQN algorithm(pytorch)
- *./fcfs.py* is a first-come-first-served approach
- *./random_search.py* does a hyperparameter search with RayTuning to find the best network configuration
- *./utils_models.py* contains the model architecture used for testing
- *./utils_nextg.py* contains code for management the Environment, Queue and Request structure
- *./utils_parameters.py* has some global definitions for imports
- *./utils_replay_buffers.py* contains a set of different replay buffers used for testing
- *./utils.py* has a few general functions that don't fit on the other categories

## Reference

If using this code for anything, please cite the following paper:

*(Waiting IEEE Release)*