import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.type_aliases import Schedule
import os
import gymnasium as gym
import copy

# Custom MlpFeatureExtractor
class CustomMlpExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, device, phase, option, prev_feature_extractor = nn.Sequential()):
        super().__init__(feature_dim, net_arch, activation_fn, device)

        # Feature extractor from Approach phase
        self.prev_policy_net = prev_feature_extractor["policy_net"].to(device)
        self.prev_value_net = prev_feature_extractor["value_net"].to(device)
        
        # Builds combined feature extractor
        last_layer = self.prev_policy_net[-1].out_features + 16
        combined_extractor = []

        for curr_layer in net_arch:
            combined_extractor.append(nn.Sequential(nn.Linear(last_layer, curr_layer)))
            combined_extractor.append(activation_fn())
            last_layer = curr_layer

        # Declares policy and value net
        self.policy_net = nn.Sequential(*combined_extractor).to(device)
        self.value_net = nn.Sequential(*copy.deepcopy(combined_extractor)).to(device)

        self.forward_actor = self.forward_actor_1
        self.forward_critic = self.forward_critic_1
     
     # Forward function for the actor
    def forward_actor_1(self, features: th.Tensor) -> th.Tensor:
        '''
        In:
            - features - torch.Tensor(batch, 7+7 ó 7+7+16): observations from the environment.

        Out:
            - latent - torch.Tensor(batch, net_arch[-1]): latent features processed from the observations. 
        '''
        
        # Process end effector and object position with previous feature extractor
        ee_obj = self.prev_policy_net(features[:, :14])

        # Return latent activations
        return self.policy_net(th.cat((ee_obj, features[:, 14:]), dim = -1))
    
    # Forward function for the critic
    def forward_critic_1(self, features: th.Tensor) -> th.Tensor:
        '''
        In:
            - features - torch.Tensor(batch, 7+7 ó 7+7+16): observations from the environment.

        Out:
            - latent - torch.Tensor(batch, net_arch[-1]): latent features processed from the observations. 
        '''
        
        # Process end effector and object position with previous feature extractor
        ee_obj = self.prev_value_net(features[:, :14])

        # Return latent activations
        return self.value_net(th.cat((ee_obj, features[:, 14:]), dim = -1))



# Define a custom policy
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            **kwargs,
        )

        features = 64
        action_shape = 2

        if net_arch != None:
            features = net_arch[-1]

        if action_space.shape != ():
            action_shape = action_space.shape[0]

        # Replace the actor with your custom network
        self.action_net = nn.Sequential(
            nn.Linear(features, action_shape),
            nn.Tanh())

        # Reinitialize parameters (important)
        self.action_net.apply(self.init_weights)
