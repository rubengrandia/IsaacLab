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
from isaaclab.utils.timer import Timer

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
    def forward(cls) -> None:
        """Reconcile body state from joint coordinates for the dirtied worlds only.

        Uses Kamino's loop-closure FK (:meth:`_forward_kamino`) when
        :attr:`KaminoSolverCfg.use_fk_solver` is enabled. The base
        :meth:`NewtonManager.forward` path calls Newton ``eval_fk``, which
        treats every articulation joint (including ``excludeFromArticulation``
        loop closures) as an independent DOF and violates kinematic constraints.

        For the Kamino (maximal-coordinate) solver, ``_forward_kamino`` runs
        ``solver.reset()``, which overwrites the authoritative ``state_0.body_q`` /
        ``body_qd``. Restricting the solve to :attr:`_world_reset_mask` keeps
        in-flight (non-reset) worlds untouched, and zeroing the masks afterwards
        means the next :meth:`step` does not redundantly re-solve them. A ``None``
        mask (before :meth:`start_simulation` allocates it) falls back to the full
        reconcile, and the mask is seeded dirty at allocation so the first call
        here establishes a kinematically consistent initial state for all worlds.
        """
        if cls._get_kamino_solver_cfg().use_fk_solver:
            cls._forward_kamino(world_mask=cls._world_reset_mask)
        else:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, cls._fk_reset_mask)
        if cls._world_reset_mask is not None:
            cls._world_reset_mask.zero_()
        if cls._fk_reset_mask is not None:
            cls._fk_reset_mask.zero_()

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
    def step(cls) -> None:
        """Step the physics simulation."""
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        # Notify solver of model changes
        if cls._model_changes:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    cls._solver.notify_model_changed(change)
                NewtonManager._model_changes = set()

        # Lazy CUDA graph capture: deferred from initialize_solver() when RTX was active.
        # By the time step() is first called, RTX has fully initialized (all cudaImportExternalMemory
        # calls are done) and is idle between render frames — giving us a clean capture window.
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device
        if cls._graph_capture_pending and cfg is not None and cfg.use_cuda_graph and "cuda" in device:  # type: ignore[union-attr]
            NewtonManager._graph_capture_pending = False
            NewtonManager._graph = cls._capture_relaxed_graph(device)
            if cls._graph is not None:
                # Kamino: StateKamino.from_newton() lazily allocates body_f_total,
                # joint_q_prev, and joint_lambdas via wp.clone/wp.zeros during the
                # first step() inside graph capture. Replay once to pin those
                # memory-pool addresses before any eager solver.reset() call.
                wp.capture_launch(cls._graph)
                logger.info("Newton CUDA graph captured (deferred relaxed mode, RTX-compatible)")
            else:
                logger.warning("Newton deferred CUDA graph capture failed; using eager execution")

        if cls._get_kamino_solver_cfg().use_fk_solver:
            cls._forward_kamino(world_mask=cls._world_reset_mask)
        else:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, cls._fk_reset_mask)

        # Zero both masks after consumption
        NewtonManager._world_reset_mask.zero_()
        NewtonManager._fk_reset_mask.zero_()

        # Step simulation (graphed or not; _graph is None when capture is disabled or failed)
        if cfg is not None and cfg.use_cuda_graph and cls._graph is not None and "cuda" in device:  # type: ignore[union-attr]
            wp.capture_launch(cls._graph)
        else:
            with wp.ScopedDevice(device):
                cls._simulate_physics_only()
        if cls._usdrt_stage is not None:
            cls._mark_transforms_dirty()

        # Launch solver-specific debug logging after stepping.
        cls._log_solver_debug()

        PhysicsManager._sim_time += cls._solver_dt * cls._num_substeps

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: KaminoSolverCfg) -> None:
        """Construct :class:`SolverKamino` and populate the base-class slots.

        Sets :attr:`NewtonManager._needs_collision_pipeline` to ``True`` only
        when ``use_collision_detector=False`` (Kamino's internal detector
        handles contacts otherwise).
        """
        NewtonManager._solver = SolverKamino(model, solver_cfg.to_solver_config())
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_collision_detector

    @classmethod
    def _capture_or_defer_cuda_graph(cls) -> None:
        """Capture the physics CUDA graph, or defer if RTX is initializing."""
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device
        use_cuda_graph = cfg is not None and cfg.use_cuda_graph and "cuda" in device  # type: ignore[union-attr]

        with Timer(name="newton_cuda_graph", msg="CUDA graph took:"):
            if not use_cuda_graph:
                NewtonManager._graph = None
                return
            if cls._usdrt_stage is None:
                # No RTX active — use standard Warp capture (cudaStreamCaptureModeGlobal).
                with wp.ScopedCapture() as capture:
                    cls._simulate_physics_only()
                NewtonManager._graph = capture.graph
                logger.info("Newton CUDA graph captured (standard Warp mode)")

                # TODO: streamline this with base NewtonManager
                # Kamino: StateKamino.from_newton() lazily allocates body_f_total,
                # joint_q_prev, and joint_lambdas via wp.clone/wp.zeros during the
                # first step() inside graph capture. Replay once to pin those
                # memory-pool addresses before any eager solver.reset() call.
                wp.capture_launch(cls._graph)
            else:
                # RTX is active during initialization — cudaImportExternalMemory and other
                # non-capturable RTX ops run on background CUDA streams right now.
                # Defer capture to the first step() call, after RTX is fully initialized
                # and idle between render frames (clean capture window).
                NewtonManager._graph = None
                NewtonManager._graph_capture_pending = True
                logger.info("Newton CUDA graph capture deferred until first step() (RTX active)")
