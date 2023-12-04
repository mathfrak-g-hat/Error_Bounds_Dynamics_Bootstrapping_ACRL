# Error_Bounds_Dynamics_Bootstrapping_ACRL
 Code used to generate the figures in the paper "Error bounds and dynamics of bootstrapping 
 in actor-critic reinforcement learning" (https://openreview.net/forum?id=QCjMJfSnYk).
 
# Figure 3.1

The Matlab script used to create the figure of spatial factors affecting temporal frequency.

# Figure C.1

Implementation of TD3 and two variants of target network updates, "small target" and "mean target". 
The implementation here is a slight modification of the code used for "Addressing Function Approximation 
Error in Actor-Critic Methods" (https://proceedings.mlr.press/v80/fujimoto18a.html), available at https://github.com/sfujim/TD3.

The results shown in the paper were obtained using Python 3.8, OpenAI Gym 0.24.0, and PyTorch 1.11 (+CUDA 11.1).


# Bibtex

```
@article{
	anonymous2023error,
	title={Error bounds and dynamics of bootstrapping in actor-critic reinforcement learning},
	author={Anonymous},
	journal={Submitted to Transactions on Machine Learning Research},
	year={2023},
	url={https://openreview.net/forum?id=QCjMJfSnYk},
	note={Under review}
}
```