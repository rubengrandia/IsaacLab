# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino Newton manager."""

from __future__ import annotations

import logging

import warp as wp
from newton import Model, eval_fk
from newton._src.solvers.kamino._src.kinematics.joints import (
    extract_actuators_state_from_joints,
)
from newton.solvers import SolverKamino

from isaaclab.physics import PhysicsManager

from .kamino_manager_cfg import KaminoSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


class NewtonKaminoManager(NewtonManager):
    """:class:`NewtonManager` specialization for the Kamino solver.

    Uses Newton's :class:`CollisionPipeline` unless
    :attr:`KaminoSolverCfg.use_collision_detector` is ``True``, in which case
    Kamino's internal collision detector handles contact generation.
    """

    @classmethod
    def _get_kamino_solver_cfg(cls) -> KaminoSolverCfg:
        cfg = PhysicsManager._cfg
        if cfg is None:
            raise RuntimeError("Physics manager is not initialized.")
        solver_cfg = getattr(cfg, "solver_cfg", None)
        if not isinstance(solver_cfg, KaminoSolverCfg):
            raise TypeError(f"Expected KaminoSolverCfg, got {type(solver_cfg).__name__}.")
        return solver_cfg

    @classmethod
    def _eval_fk(cls, world_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        """Reconcile body state from joint coordinates for the Kamino solver.

        Uses Kamino's loop-closure FK (:meth:`_forward_kamino`) when
        :attr:`KaminoSolverCfg.use_fk_solver` is enabled. The base ``eval_fk`` path treats
        every articulation joint (including ``excludeFromArticulation`` loop closures) as an
        independent DOF and violates kinematic constraints.

        For the Kamino (maximal-coordinate) solver, ``_forward_kamino`` runs ``solver.reset()``,
        which overwrites the authoritative ``state_0.body_q`` / ``body_qd``. Restricting the
        solve to ``world_mask`` keeps in-flight (non-reset) worlds untouched; a ``None``
        ``world_mask`` reconciles all worlds (the full-reconcile path used by
        :meth:`forward` when :attr:`_forward_full_reconcile` is set, and at initial setup).

        Args:
            world_mask: Per-world mask of worlds to reconcile (``None`` means all).
            fk_mask: Per-articulation mask, used only on the non-``use_fk_solver`` ``eval_fk``
                fallback.
        """
        if cls._get_kamino_solver_cfg().use_fk_solver:
            cls._forward_kamino(world_mask=world_mask)
        else:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, fk_mask)

    @classmethod
    def _forward_kamino(cls, world_mask: wp.array | None = None) -> None:
        """Kamino-specific forward kinematics via ``solver.reset()``.

        Used when :attr:`KaminoSolverCfg.use_fk_solver` is ``True`` (default). Extracts actuated
        coordinates from Newton ``joint_q`` / ``joint_qd`` and passes ``actuator_q`` /
        ``actuator_u`` so Kamino FK resolves passive joints.

        Args:
            world_mask: Per-world mask indicating which worlds to reset.
                Shape ``(num_worlds,)``, dtype ``wp.int32``. If None, resets all worlds.
        """

        solver_kamino = cls._solver._solver_kamino
        model_kamino = cls._solver._model_kamino
        effective_world_mask = world_mask if world_mask is not None else solver_kamino._all_worlds_mask

        extract_actuators_state_from_joints(
            model=model_kamino,
            world_mask=effective_world_mask,
            joint_q=cls._state_0.joint_q,
            joint_u=cls._state_0.joint_qd,
            actuator_q=solver_kamino._actuators_q,
            actuator_u=solver_kamino._actuators_u,
        )

        cls._solver.reset(
            state_out=cls._state_0,
            actuator_q=solver_kamino._actuators_q,
            actuator_u=solver_kamino._actuators_u,
            world_mask=world_mask,
        )

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: KaminoSolverCfg) -> None:
        """Construct :class:`SolverKamino` and populate the base-class slots.

        Sets :attr:`NewtonManager._needs_collision_pipeline` to ``True`` only
        when ``use_collision_detector=False`` (Kamino's internal detector
        handles contacts otherwise).

        Configures the shared FK-reconcile flags for the Kamino (maximal-coordinate)
        solver: it always reconciles body state from joints before stepping
        (:attr:`NewtonManager._reconcile_fk_before_step`), and :meth:`forward` reconciles
        only the dirtied worlds (:attr:`NewtonManager._forward_full_reconcile` is ``False``)
        because ``solver.reset()`` is destructive to in-flight (non-reset) worlds.
        """
        NewtonManager._solver = SolverKamino(model, solver_cfg.to_solver_config())
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_collision_detector
        NewtonManager._reconcile_fk_before_step = True
        NewtonManager._forward_full_reconcile = False
