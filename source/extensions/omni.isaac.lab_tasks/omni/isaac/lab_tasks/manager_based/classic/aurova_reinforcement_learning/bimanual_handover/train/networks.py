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
        # SB3 built its optimizer around the original action head in super().
        # Rebuild it so the replacement head is actually trained.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


# Applies ONE shared-weight MLP trunk to each arm's self-centric observation
# block instead of one trunk over the whole flat observation. This is the
# actual "two arms learn from each other" mechanism for the bidirectional
# handover task (bimanual_bidirectional_env.py): GEN3's phase-0 experience and
# UR5e's phase-1 experience both update the same weights, because both arms'
# blocks are pushed through the identical nn.Sequential. Both arms' latents
# are concatenated for the critic, giving a single centralized value estimate
# that matches the single team-reward scalar the env returns.
#
# Duck-types SB3's MlpExtractor interface (forward_actor/forward_critic/
# forward + latent_dim_pi/latent_dim_vf attributes) rather than subclassing
# it, since ActorCriticPolicy only ever calls through that interface -- same
# trick CustomMlpExtractor above uses, just without inheriting a constructor
# shaped for a single flat trunk.
class SharedArmMlpExtractor(nn.Module):
    def __init__(self, feature_dim, arm_obs_dim, net_arch, activation_fn, device):
        super().__init__()

        if feature_dim != 2 * arm_obs_dim:
            raise ValueError(
                f"SharedArmMlpExtractor expects obs = [UR5e block | GEN3 block] of "
                f"{arm_obs_dim} each ({2 * arm_obs_dim} total), got feature_dim={feature_dim}. "
                "Check BimanualBidirectionalCfg.arm_obs_dim against the observation "
                "layout in BimanualBidirectional._get_observations."
            )

        self.arm_obs_dim = arm_obs_dim
        net_arch = list(net_arch) if net_arch else [64]

        def build_trunk():
            layers, last = [], arm_obs_dim
            for width in net_arch:
                layers += [nn.Linear(last, width), activation_fn()]
                last = width
            return nn.Sequential(*layers).to(device)

        # Separate actor/critic trunks (mirrors the policy_net/value_net split
        # every extractor in this file uses), but each is itself shared
        # between the two arms.
        self.policy_net = build_trunk()
        self.value_net = build_trunk()

        hidden = net_arch[-1]
        self.latent_dim_pi = 2 * hidden
        self.latent_dim_vf = 2 * hidden

    def _split(self, features: th.Tensor):
        return features[:, : self.arm_obs_dim], features[:, self.arm_obs_dim :]

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        ur5e_block, gen3_block = self._split(features)
        return th.cat((self.policy_net(ur5e_block), self.policy_net(gen3_block)), dim=-1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        ur5e_block, gen3_block = self._split(features)
        return th.cat((self.value_net(ur5e_block), self.value_net(gen3_block)), dim=-1)

    def forward(self, features: th.Tensor):
        return self.forward_actor(features), self.forward_critic(features)


# Turns the concatenated [ur5e_latent | gen3_latent] actor latent back into
# two per-arm action vectors via ONE shared, weight-tied Linear+Tanh head
# (mirrors CustomActorCriticPolicy's action_net above, applied per-arm instead
# of once). Plugged in as SharedArmActorCriticPolicy.action_net, so SB3's
# existing action-distribution/SDE machinery (unchanged) just calls this like
# any other action_net: mean_actions = self.action_net(latent_pi).
class SharedArmActionHead(nn.Module):
    def __init__(self, hidden_dim, action_dim_per_arm, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head = nn.Sequential(nn.Linear(hidden_dim, action_dim_per_arm), nn.Tanh()).to(device)

    def forward(self, latent_pi: th.Tensor) -> th.Tensor:
        ur5e_latent = latent_pi[:, : self.hidden_dim]
        gen3_latent = latent_pi[:, self.hidden_dim :]
        return th.cat((self.head(ur5e_latent), self.head(gen3_latent)), dim=-1)


# Custom policy for Isaac-Bimanual-Bidirectional-v0 (see
# bimanual_bidirectional_env.py). Same override strategy as
# CustomActorCriticPolicy (swap action_net after super().__init__()), plus a
# custom mlp_extractor for the actual weight-sharing. arm_obs_dim and
# action_dim_per_arm come from policy_kwargs in
# agents/sb3_ppo_cfg_bidirectional.yaml and must match
# BimanualBidirectionalCfg.arm_obs_dim / the env's per-arm action width.
#
# Biggest untested-in-this-sandbox risk: interaction with use_sde=True
# (log_std/latent_sde sizing is built by ActorCriticPolicy._build() from
# mlp_extractor.latent_dim_pi *before* action_net is swapped below -- as long
# as latent_dim_pi truthfully reflects what forward_actor() returns, which it
# does here, that sizing should stay consistent, matching how
# CustomActorCriticPolicy already overrides action_net post-hoc under the same
# use_sde=True config -- but this hasn't been run against a real SB3/torch
# install to confirm).
class SharedArmActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None,
                 arm_obs_dim=None, action_dim_per_arm=9, **kwargs):

        self._arm_obs_dim = arm_obs_dim if arm_obs_dim is not None else observation_space.shape[0] // 2
        self._action_dim_per_arm = action_dim_per_arm
        self._shared_net_arch = list(net_arch) if net_arch else [256, 128, 64]

        super(SharedArmActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            **kwargs,
        )

        hidden = self._shared_net_arch[-1]
        self.action_net = SharedArmActionHead(
            hidden_dim=hidden, action_dim_per_arm=self._action_dim_per_arm, device=self.device
        )
        self.action_net.apply(self.init_weights)

        # ActorCriticPolicy.__init__ already built an optimizer containing the
        # discarded default head. Register the shared head before training.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SharedArmMlpExtractor(
            feature_dim=self.features_dim,
            arm_obs_dim=self._arm_obs_dim,
            net_arch=self._shared_net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
