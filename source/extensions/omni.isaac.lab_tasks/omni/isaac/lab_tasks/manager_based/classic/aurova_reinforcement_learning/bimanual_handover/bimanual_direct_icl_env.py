from __future__ import annotations

import torch
from collections.abc import Sequence

from .bimanual_direct_env import BimanualDirect
from .bimanual_direct_icl_env_cfg import BimanualDirectICLCfg

'''
Baseline + inverse curriculum. Thin subclass: the only behaviour added is
rescaling the holder's reset-pose sampling range according to a curriculum
level, plus the bookkeeping to move that level up and down.

Everything else -- reward, observations, terminations, the receiving arm's
control -- is BimanualDirect's, untouched, so a run of this task at its last
level is the baseline.
'''


class BimanualDirectICL(BimanualDirect):
    cfg: BimanualDirectICLCfg

    def __init__(self, cfg: BimanualDirectICLCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.icl_level = 0
        self._icl_ep_count = 0
        self._icl_succ_count = 0
        self._icl_eps_at_level = 0
        self._icl_steps_at_level = 0

        # ee_pose_ranges is [num_robots, 6, 2]: per DOF, [lo, hi] for the
        # uniform sample drawn at reset. The parent builds it once in __init__
        # as nominal +/- apply_range*ee_pose_incs. Keep the full-range version
        # and the nominal centre so a level can be rebuilt from scratch each
        # time rather than repeatedly rescaled (which would compound).
        self._icl_full_ranges = self.ee_pose_ranges.clone()
        self._icl_nominal = self.ee_pose_ranges.mean(dim=-1, keepdim=True)  # [R, 6, 1]
        self._apply_icl_level()

    def _apply_icl_level(self):
        '''
        Rebuild ee_pose_ranges for the current level: shrink each DOF's sampling
        interval toward its nominal centre by the level's scale factor.

        scale 0.0 -> lo == hi == nominal, i.e. the object is presented in
        exactly the same pose every episode.
        scale 1.0 -> the untouched baseline range.
        '''
        if not self.cfg.use_icl:
            return
        level = min(self.icl_level, len(self.cfg.icl_range_scale) - 1)
        scale = float(self.cfg.icl_range_scale[level])
        self.ee_pose_ranges = self._icl_nominal + scale * (self._icl_full_ranges - self._icl_nominal)

    def _update_icl(self, success: torch.Tensor, done: torch.Tensor):
        if not self.cfg.use_icl:
            return
        # Counted before any early return, so it accumulates unconditionally.
        self._icl_steps_at_level += self.num_envs
        n_done = int(done.sum().item())
        if n_done == 0:
            return
        self._icl_ep_count += n_done
        self._icl_eps_at_level += n_done
        self._icl_succ_count += int((success & done).sum().item())

        max_level = len(self.cfg.icl_range_scale) - 1

        # Forced promotion: guarantees the curriculum actually traverses its
        # levels even if the success threshold is never met. Counted in
        # environment steps rather than episodes -- the episode-counted version
        # (job 4228865) never fired despite ~660k completed episodes against a
        # 20k threshold, and icl_level sat at 0 for 131M steps.
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

    def _set_level(self, level: int):
        '''Change level and reset every counter in one place, so a future edit
        cannot leave one of them stale.'''
        self.icl_level = level
        self._apply_icl_level()
        self._icl_eps_at_level = 0
        self._icl_steps_at_level = 0
        self._icl_ep_count = 0
        self._icl_succ_count = 0

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated, terminated = super()._get_dones()

        # Same success definition the parent logs: reached the target and did
        # not end badly.
        success = self.obj_reached_target & ~truncated
        self._update_icl(success, torch.logical_or(truncated, terminated))

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["metrics/icl_level"] = float(self.icl_level)
        self.extras["log"]["metrics/icl_eps_at_level"] = float(self._icl_eps_at_level)
        self.extras["log"]["metrics/icl_steps_at_level"] = float(self._icl_steps_at_level)
        self.extras["log"]["metrics/icl_scale"] = float(
            self.cfg.icl_range_scale[min(self.icl_level, len(self.cfg.icl_range_scale) - 1)]
        )
        # Spread of the object's position across environments. This is the
        # self-check that the curriculum is actually doing something: it should
        # be ~0 at level 0 and grow with the level. If it stays flat, the range
        # rescaling is not reaching the sampler and the run is meaningless --
        # which is exactly the failure mode that went unnoticed in the
        # self-play curriculum until a dedicated diagnostic caught it.
        # MUST be env-local. Measured in world coordinates this is dominated by
        # the environment grid (256 envs at 2.5m spacing -> std ~7.7m), which
        # swamps the +/-0.15m the curriculum actually varies. Job 4205270
        # logged exactly that useless 7.7 before this was corrected.
        obj_pos = (
            self.scene.rigid_objects["object"].data.body_state_w[:, 0, :3] - self.scene.env_origins
        )
        self.extras["log"]["metrics/obj_pos_spread"] = obj_pos.std(dim=0).mean().item()

        return truncated, terminated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # The parent samples from self.ee_pose_ranges, so simply having
        # rebuilt that tensor is enough -- no placement code of our own.
        super()._reset_idx(env_ids)
