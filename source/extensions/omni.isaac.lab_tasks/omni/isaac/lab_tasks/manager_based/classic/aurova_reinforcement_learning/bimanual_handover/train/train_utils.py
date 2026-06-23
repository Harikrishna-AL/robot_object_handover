import gymnasium as gym
import numpy as np
import torch

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Class for overriding and adding noise to the observations in the environment ---
class AddNoiseObservation(gym.ObservationWrapper):
    def __init__(self, env, noise_std = 0.1):
        super(AddNoiseObservation, self).__init__(env)
        self.noise_std = noise_std      # Standard deviation for noise adding

    # Add Gaussian noise to the observation --> Overrides method of GYM
    def observation(self, observation):
        '''
        In:
            - observation - dict: dictionary of observations. The key "policy" contains a tensor, ...
                ... which corresponds to the observation
                
        Out:
            - dict: dictionary where the key is "policy" and contains the noised observation.
        '''

        return {"policy": torch.normal(mean = observation["policy"], std = self.noise_std)}


# define models (stochastic and deterministic models) using mixins
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.action_layer(x)), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}