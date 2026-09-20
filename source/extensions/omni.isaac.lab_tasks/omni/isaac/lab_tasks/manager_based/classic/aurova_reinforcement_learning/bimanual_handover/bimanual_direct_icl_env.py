from __future__ import annotations

import torch
from collections.abc import Sequence

from .bimanual_direct_env import BimanualDirect
from .bimanual_direct_icl_env_cfg import BimanualDirectICLCfg

'''
Baseline + inverse curriculum on the receiver's STARTING DISTANCE to the object.

Thin subclass: the only added behaviour is overwriting the receiving arm's
joint state at reset with an interpolation between its home configuration and a
precomputed on-object configuration, plus the bookkeeping to advance the level.
Reward, observations, terminations and the object's randomisation are
BimanualDirect's, untouched -- so this task at its final level IS the baseline.
'''


class BimanualDirectICL(BimanualDirect):
    cfg: BimanualDirectICLCfg

    def __init__(self, cfg: BimanualDirectICLCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.icl_level = 0
        self._icl_ep_count = 0
        self._icl_succ_count = 0
        self._icl_steps_at_level = 0

        R = self.cfg.GEN3
        self._icl_n_arm = len(self.cfg.joints[R])
        self._icl_q_near = torch.tensor(
            list(self.cfg.icl_near_object_arm_q), dtype=torch.float32, device=self.device
        )
        assert self._icl_q_near.shape[0] == self._icl_n_arm, (
            f"icl_near_object_arm_q has {self._icl_q_near.shape[0]} values, "
            f"receiver arm has {self._icl_n_arm} joints"
        )

    # ------------------------------------------------------------------
    def _reach_frac(self) -> float:
        lvl = min(self.icl_level, len(self.cfg.icl_reach_frac) - 1)
        return float(self.cfg.icl_reach_frac[lvl])

    def _set_level(self, level: int):
        '''Change level and clear every counter in one place, so a future edit
        cannot leave one of them stale.'''
        self.icl_level = level
        self._icl_ep_count = 0
        self._icl_succ_count = 0
        self._icl_steps_at_level = 0

    def _update_icl(self, success: torch.Tensor, done: torch.Tensor):
        if not self.cfg.use_icl:
            return
        self._icl_steps_at_level += self.num_envs   # before any early return
        n_done = int(done.sum().item())
        if n_done == 0:
            return
        self._icl_ep_count += n_done
        self._icl_succ_count += int((success & done).sum().item())

        max_level = len(self.cfg.icl_reach_frac) - 1
        if self._icl_steps_at_level >= self.cfg.icl_force_promote_steps and self.icl_level < max_level:
            self._set_level(self.icl_level + 1)
            return
        if self._icl_ep_count < self.cfg.icl_window:
            return

        rate = self._icl_succ_count / max(self._icl_ep_count, 1)
        if rate >= self.cfg.icl_promote_rate and self.icl_level < max_level:
            self._set_level(self.icl_level + 1)
        elif rate < self.cfg.icl_demote_rate and self.icl_level > 0:
            self._set_level(self.icl_level - 1)
        else:
            self._icl_ep_count = 0
            self._icl_succ_count = 0

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Parent places the object (randomised, at full strength) and both arms.
        super()._reset_idx(env_ids)
        if not self.cfg.use_icl:
            return
        frac = self._reach_frac()
        if frac <= 0.0:
            return  # final level == the unmodified baseline reset

        R = self.cfg.GEN3
        key = self.cfg.keys[R]
        art = self.scene.articulations[key]
        n = self._icl_n_arm

        # Interpolate in JOINT space between the home configuration the parent
        # just wrote and the precomputed on-object configuration. No IK at
        # reset: _reset_idx runs outside the physics loop, so it cannot step the
        # sim to refresh link transforms, and a single differential IK step is
        # nowhere near enough for a ~0.4m target (measured: ~5cm of movement).
        #
        # The on-object configuration was solved for the object's NOMINAL pose,
        # so with the object randomised +/-0.15m the arm lands approximately
        # near it rather than exactly on it. That is intentional -- the point is
        # to shorten the reach while KEEPING the variation that the previous
        # curriculum destroyed.
        q_home = art.data.joint_pos.clone()
        q_new = q_home.clone()
        q_new[:, :n] = q_home[:, :n] + frac * (self._icl_q_near.unsqueeze(0) - q_home[:, :n])

        joint_vel = torch.zeros_like(q_new)
        art.write_joint_state_to_sim(q_new[env_ids], joint_vel[env_ids], None, env_ids)
        # Keep the running IK/action targets consistent, or the first
        # perform_increment drives the arm straight back to the home pose.
        self.reset_joint_positions[R][env_ids] = q_new[env_ids]
        self.actions[R][env_ids] = q_new[env_ids]

    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated, terminated = super()._get_dones()

        success = self.obj_reached_target & ~truncated
        self._update_icl(success, torch.logical_or(truncated, terminated))

        if "log" not in self.extras:
            self.extras["log"] = dict()
        log = self.extras["log"]
        log["metrics/icl_level"] = float(self.icl_level)
        log["metrics/icl_reach_frac"] = self._reach_frac()
        log["metrics/icl_steps_at_level"] = float(self._icl_steps_at_level)

        # Self-check, env-local and unconfoundable: how far the receiver's
        # fingertips actually are from the object. Must be small at level 0 and
        # grow with the level. The previous curriculum's self-check measured
        # world-frame spread, which the 2.5m environment grid swamped (~7.7m),
        # making a working curriculum indistinguishable from a no-op.
        R = self.cfg.GEN3
        tips = self.scene.articulations[self.cfg.keys[R]].data.body_state_w[:, self.finger_tips[R], 0:3].mean(1)
        obj = self.scene.rigid_objects["object"].data.body_state_w[:, 0, :3]
        log["metrics/icl_tips_to_obj"] = torch.norm(tips - obj, dim=-1).mean().item()

        return truncated, terminated
