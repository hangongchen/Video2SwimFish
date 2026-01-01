# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the salmon "swim to a target in water" RL task.

Mirrors the Template-Fish-Direct-v0 target-reaching pipeline, but:
  * uses the salmon skeleton (deformable deactivated for speed),
  * applies an analytic MuJoCo ellipsoid hydrodynamic wrench each physics step
    (the "water" -- gravity is zeroed, so the fish is neutrally buoyant and must
    undulate to make headway),
  * drives the D6 joints by PD POSITION targets (the env authors the missing
    angular DriveAPI so set_joint_position_target actuates them).
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils import configclass


ASSETS_DIR = Path(__file__).resolve().parent / "agents"
SALMON_STAGE_PATH = ASSETS_DIR / "salmon" / "uniformed_scale_small_salmon.usd"

# Swim PD-position gains on the (DriveAPI-enabled) D6 joints.
# Position-mode D6 drive gains. PATH 1: revert to the GENTLE config that ran 4.5 h with the
# soft body WITHOUT blowing up (stiffness 60 / damping 6 / armature 0.01 / +/-15 deg / action
# 0.22). Aggressive gains + the FEM soft body are numerically incompatible -- every aggressive
# setting self-diverged and damping/dt only DELAYED it (blow-up step 684 -> 1000 -> 1380, never
# eliminated). So stay in the stable regime and fix THRUST via the reward/aiming, not bigger gains.
SWIM_STIFFNESS = 60.0
SWIM_DAMPING = 6.0
SWIM_ARMATURE = 0.01

# Physics timestep. 960 Hz (dt=1/960) -- the SETTLED stable operating point. dt-search is DONE: the
# FEM blow-up is a CFL instability, and the no-blow-up boundary sits just below 1/720. Empirics (each
# blows up only AFTER the gait coheres, ~epoch 4-6, not early): 1/240, 1/480, 1/720 all BLEW UP;
# 1/960 (1.04 ms) is STABLE (64/64 envs ran full 1500-step episodes, 0 blow-ups). So dt_crit is in
# (1.04, 1.39] ms and 1/960 is the fastest integer-decimation rate with enough margin -- do NOT
# re-probe 1/600..1/900, they're inside the unstable band or too close. softening/damping change the
# unstable mode's SIGN but the explicit CFL ceiling still needs this small dt -- complementary, not a
# substitute. Defined ONCE, used for BOTH SimulationCfg.dt AND episode_length_s so the episode length
# can never silently desync from the physics rate.
SIM_DT = 1.0 / 960


SALMON_SWIM_ARTICULATION_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/skeleton",
    articulation_root_prim_path=None,
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SALMON_STAGE_PATH),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=20.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
    ),
    # Level heading; the salmon spine lies along its body axis.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        rot=(0.5, -0.5, 0.5, -0.5),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=1e5,    # gentle gains (60 stiffness, +/-15 deg) cap torque at ~15.7 N.m
                                     # anyway, so unbounded is fine here (matches the 4.5h-stable run).
            armature=SWIM_ARMATURE,
            stiffness=SWIM_STIFFNESS,
            damping=SWIM_DAMPING,
        )
    },
)


TARGET_MARKER_CFG = sim_utils.CuboidCfg(
    size=(0.15, 0.15, 0.15),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
)


@configclass
class SalmonSwimEnvCfg(DirectRLEnvCfg):
    # env timing (back to the 4.5h-stable 240 Hz; gentle gains don't need 480 Hz, and 240 is 2x faster)
    # FEM hardening (b): physics at 480 Hz (dt=1/480) for finer integration -> raises the stable
    # frequency, suppressing the FEM mode that diverges. decimation 8->16 keeps CONTROL at 30 Hz.
    decimation = 32   # dt 1/960 * decimation 32 -> control still at exactly 30 Hz
    # dt-CORRECT episode length: 1500 control steps * (decimation * SIM_DT) control period = 50 s.
    # (The old hard-coded /480 form would give 100 s at decimation=32 -> wrong; use SIM_DT.)
    episode_length_s = 1500 * decimation * SIM_DT  # ~50 s -> max_episode_length = 1500 control steps
    action_space = 1   # placeholder, resolved at runtime
    observation_space = 1
    state_space = 0

    # "in water" -> neutrally buoyant: zero gravity, analytic drag does the rest.
    # enable_external_forces_every_iteration is required for set_external_force_and_torque.
    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT,         # 960 Hz (settled stable operating point; 1/720, 1/480, 1/240 all blew up)
        render_interval=decimation,
        gravity=(0.0, 0.0, 0.0),
        physx=PhysxCfg(
            solver_type=1,
            enable_external_forces_every_iteration=True,
            # GPU buffers sized for the FEM deformable (needed when with_deformable=True;
            # harmless otherwise). Mirrors the salmon_IL task.
            gpu_collision_stack_size=2 ** 30,
            gpu_heap_capacity=2 ** 30,
            gpu_temp_buffer_capacity=2 ** 28,
            gpu_max_soft_body_contacts=2 ** 22,
            gpu_max_particle_contacts=2 ** 28,
        ),
    )

    robot_cfg: ArticulationCfg = SALMON_SWIM_ARTICULATION_CFG

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=8.0,
        replicate_physics=False,  # salmon USD has attachments PhysX cannot replicate
    )

    # Keep the FEM deformable body ACTIVE. Off by default: RL trains the skeleton (the
    # hydro force is computed on the bones anyway) and FEM x many-envs is heavy + the
    # soft-body attachments are fragile under cloning. Turn ON for small-env
    # visualization of the trained policy (auto-fixes the deformable collision offsets).
    with_deformable = False
    # FEM material Young's modulus (Pa) override, applied to the deformable in _prepare_env_assets
    # when with_deformable. Default Isaac value is ~5e7 (50 MPa, stiff) -> diverges. 1e5 (0.1 MPa,
    # very soft) is the strongest stability test; raise toward realism if it holds. None = leave as-is.
    fem_youngs_modulus = 1.0e5
    # FEM-CURE material DAMPING, authored on the SAME material prim as youngsModulus (in
    # _prepare_env_assets). elasticityDamping is STIFFNESS-PROPORTIONAL (Rayleigh-beta): its modal
    # damping ratio grows with frequency (~beta*omega), so it bites hardest on exactly the stiff
    # high-omega FEM mode that goes CFL-unstable -> drives that mode's growth rate <= 0. THIS is the
    # sign-flip that turns (b)'s DELAY (blow-up step 104 -> 1085) into a cure. Schema default 0.005;
    # 0.05 is 10x but still <<1 (the documented safe regime). dampingScale in [0,1] sets damping
    # authority (1.0 = full). None = leave the asset value. MUST land on the material prim (a material
    # attr written to the /deformable_salmon BODY prim is a SILENT no-op).
    fem_elasticity_damping = 0.05
    fem_damping_scale = 1.0
    # INFERENCE option: on reset, teleport the FEM soft body WITH the bones (rigidly place its
    # rest shape at the new root pose + zero nodal velocities). Without this the soft mesh is
    # left behind on a reset -> it snaps back (visible "explosion" on the success reset during
    # play). Off in training (the snap is invisible/harmless there); the play script turns it on.
    reset_deformable = False
    # Substring that identifies the FEM deformable subtree in the spawned USD. The salmon asset
    # names it `deformable_salmon`; other fish USDs (e.g. the SimFishLib zebrafish) name the
    # deformable mesh differently (`final_mesh`). Every deformable-locating scan in the env keys
    # off this token, so a different fish only needs to override it (not edit the env).
    deformable_prim_token = "deformable_salmon"
    # Some fish USDs (SimFishLib video->USD pipeline) apply PhysicsArticulationRootAPI to EVERY bone,
    # which Isaac rejects ("Failed to find a single articulation"). When True, the env strips those
    # per-bone roots and applies one root on the spawn prim. Salmon ships a correct single root -> off.
    fix_multi_articulation_root = False
    # SimFishLib deformable meshes carry a NEGATIVE-scale (reflected, det<0) xform from the Blender
    # export, which Isaac's DeformablePrim rejects. When True the env flips it to a right-handed frame
    # (positive scale + negated points + reversed winding, same world geometry). Salmon -> off.
    fix_reflected_deformable = False
    # Keep the rigid-bone colliders ENABLED (default disables them for zero-g water with no ground
    # contact). The tank task sets this True so the bones collide with the four walls.
    keep_bone_colliders = False
    # INFERENCE option: draw the target as a translucent red sphere of radius=success_radius
    # (the actual success region) instead of the small red cuboid marker.
    marker_success_sphere = False
    # How see-through that sphere is (0 = invisible, 1 = solid). RTX treats UsdPreviewSurface
    # `opacity` as CUTOUT opacity, so the env also flips /rtx/raytracing/fractionalCutoutOpacity
    # on -- without that render setting a 0.3 sphere renders SOLID and hides the fish inside it.
    marker_success_opacity = 0.3
    # Chase the fish with the viewport camera instead of the fixed world eye/lookat in `viewer`.
    # Needed for play: a fixed camera loses the fish the moment it swims a couple of metres (the
    # deploy lift-launch sends it tens of metres away and the viewport just goes black).
    # Hydra rejects env.viewer.* on the CLI, so this plain bool is the override hook -- it rewrites
    # cfg.viewer BEFORE DirectRLEnv reads it.
    camera_follow = False
    # eye offset (metres) from the fish root when camera_follow is on. Bigger = zoomed out.
    camera_follow_eye = (1.2, 1.2, 0.6)
    # True  = camera GLUED to the fish root. The fish stays dead centre, so you can study the gait
    #         but TRANSLATION IS INVISIBLE -- it looks like a treadmill.
    # False = camera pinned to env_0's ORIGIN. The fish swims across a stationary frame, which is
    #         the only way to actually see it travel. Use with a top-down eye, e.g.
    #         camera_follow_eye=[0.0,-1.5,5.0].
    camera_follow_track = True
    # Material for the success sphere.
    # "preview" = UsdPreviewSurface + `opacity`. RTX reads that as CUTOUT opacity, so the env also
    #             switches /rtx/raytracing/fractionalCutoutOpacity on, which its own tooltip
    #             describes as "a translucency-like effect similar to alpha-blending". USE THIS.
    # "glass"   = OmniGlass MDL. Physically see-through, but MEASURED INVISIBLE here: the water
    #             scene is a black void lit only by the camera light, so clear glass has nothing
    #             to refract or reflect and renders as zero pixels (verified: 0 red pixels in the
    #             3D view across 3 frames). Do not use it in this scene.
    marker_success_material = "preview"
    # draw the red target marker at all. The IL-water task has no target point, so it sets
    # this False to keep the scene clean.
    draw_target_marker = True

    # ---- control ----
    # "position" = PD position-target actions on the D6 joints (authors the DriveAPI);
    # "effort"   = torque actions (fallback, no DriveAPI needed).
    control_mode = "position"
    # per-step terminal debug: reward terms, action stats, success flags, done/reset reasons,
    # and joint-vel max for env 0 (set env.debug_print=true). Off by default (noisy).
    debug_print = False
    # When enabled, expose pre-reset per-step tensors through extras["debug_step"] so play/debug
    # scripts can log the true terminal state instead of the post-reset buffers returned by env.step().
    step_debug_extras = False
    # position mode: action in [-1,1] -> joint target angle = action * pos_action_scale (rad)
    # 0.22 ~= the native +/-15 deg (0.262 rad) joint limit (the 4.5h-stable value).
    pos_action_scale = 0.22
    # effort mode: torque = action * action_scale (effort mode only; unused in position mode)
    action_scale = 5.0
    # D6 rot limit half-range (deg). Asset ships +/-15; widened for bigger undulation
    # amplitude -- the real thrust ceiling (mujoco_fluid_wrench thrust ~ lateral_vel^2, and
    # lateral_vel is set by how far/fast the body bends). Authored onto every D6 rotX/Y/Z
    # LimitAPI in _prepare_env_assets BEFORE clone, so the articulation's
    # soft_joint_pos_limits (which clamp the pos targets) pick up the wider range.
    joint_limit_deg = 15.0   # native asset limit (the 4.5h-stable value)

    # ---- analytic MuJoCo ellipsoid hydro (the "water") ----
    # PER-SLICE ellipsoids: each bone's drag is sized by the body slice around it (fit to
    # the FEM node cloud offline by scripts/precompute_per_bone_hydro.py), NOT the thin
    # bone -- so the undulation produces real thrust. Falls back to inertia-derived thin
    # ellipsoids if the file is missing.
    per_bone_hydro_path: str = str(ASSETS_DIR / "salmon" / "per_bone_hydro.npz")
    rho = 1000.0          # fluid density (kg/m^3)
    visc = 0.0            # Stokes viscosity (0 -> quadratic drag only)
    cd_blunt = 1.0        # normal/pressure drag
    cd_slender = 0.2      # tangential/skin drag
    cd_angular = 1.5      # rotational drag
    ck = 1.0              # Kutta lift
    cm = 1.0              # Magnus lift
    hydro_force_clip = 20.0   # per-bone FORCE clamp (N)
    # separate TORQUE clamp (N.m). Bones have tiny rotational inertia, so at coarse dt a torque at the
    # 20 N.m force-clip level imparts thousands of rad/s in ONE substep -> explicit-drag sign-flip
    # explosion (observed: jvel 31k at step 1, all envs, per-slice hydro @ 1/120). None = use
    # hydro_force_clip (back-compat for the 1/960 salmon tasks).
    hydro_torque_clip = None
    # scale factor on the loaded per-slice hydro semi-axes (drag area ~ scale^2). 1.0 = as fitted.
    hydro_semi_scale = 1.0
    # body length (m) used ONLY to log swim speed in body-lengths/s (real fish cruise 1-4 BL/s).
    # 0.0 = auto-measure from the asset mesh (panel hydro assets); set explicitly for others.
    body_length = 0.0
    # Drive ONLY the joints whose name ends with this suffix (e.g. ":1" = one D6 rotation axis).
    # None = drive all. Family ":1" alone was MEASURED at +/-0.53 BL/s with a scripted wave.
    # NOTE: "" not None -- @configclass types the field from its default, and a None default
    # makes hydra reject a string override ("Expected NoneType, Received str").
    control_dof_suffix = ""
    # Widen the D6 rotation limits past the asset's own (deg). 0.0 = keep the asset value.
    # pos_action_scale MUST be raised to match (limit in radians) or nothing changes.
    joint_limit_deg = 0.0

    debug_logs = False

    @configclass
    class ObservationScalesCfg:
        joint_pos = 1.0
        joint_vel = 0.05
        root_lin_vel = 0.5
        root_ang_vel = 0.2

    obs_scales: ObservationScalesCfg = ObservationScalesCfg()

    @configclass
    class RewardScalesCfg:
        distance = 100.0      # progress toward target (prev_dist - dist) * scale
                              # 10 -> 100: progress was drowned out by heading (0.5/step,
                              # obtainable by just facing the target). 10x makes per-step
                              # progress competitive at a modest swim speed -> reward now
                              # pulls the policy to actually SWIM, not just turn.
        success = 200.0       # one-shot bonus on reaching the target
        heading = 0.0         # DISABLED (was 0.5) -- "face the target" was a cheap local optimum
                              # (turn in place, don't swim). Only progress+success drive it now.
        time = 0.001          # small per-step time penalty (runs for the WHOLE episode)
        prereach_time = 0.0   # PRE-REACH time penalty: charged every step until this env reaches its
                              # FIRST target, then latched OFF for the rest of the episode
                              # (self._first_reach_done). Targets the exact failure we have: a fish
                              # that never scores pays for every lazy step, while a fish that is
                              # already scoring pays nothing -- so it presses on the stuck case only.
                              # SIZING (1800-step episode, success bonus 200, observed returns 130-400):
                              #   w=0.05 -> never-reach costs -90, reach@step600 costs -30,
                              #             so scoring early is worth +60 on top of the 200 bonus.
                              #   w=0.20 -> never-reach costs -360, which SWAMPS the whole return.
                              # ESCAPE RISK: out_of_bounds (root_position_limit, 20 m) ENDS the episode,
                              # so a large w can pay the fish to FLEE rather than score. At today's
                              # 0.01-0.05 BL/s that is out of reach in 60 s, but the documented
                              # lift-launch artifact reaches 3.9 m/s = 20 m in ~5 s. Keep w <= 0.05 and
                              # watch `frac_prereach` together with the episode length.
        effort = 0.0          # optional action penalty
        spin = 0.0            # anti-spin: penalty on root angular-velocity magnitude (rad/s)^2.
                              # discourages the "move by rotating the whole body" cheat. >0 to enable.
        upright = 0.0         # optional keep-level: penalty on body up-axis tilting away from world-up.
        z_band = 0.0          # DEPTH-BAND penalty: free within +/-z_band_halfwidth of the TARGET's
                              # height, linear outside. Zero-gravity water + the thin-water drag fix
                              # leave nothing to restore depth, so a slight nose-up pitch makes the
                              # fish climb out of the target plane and never return. >0 to enable.
        launch = 0.0          # ANTI-LAUNCH: penalty on linear-speed^2 weighted UP near the target
                              # (discourages the fling/overshoot that flings the fish away at the goal).
        approach = 0.0        # POTENTIAL-BASED approach shaping weight. Reward = w*(exp(-d/scale) -
                              # exp(-d_prev/scale)); steep near the target, telescoping (no hover trap).
                              # Set >0 (e.g. 30) to fill the orbit's "comfortable valley" and pull the
                              # final metre. Pair with heading=0 (heading FUNDS the orbit).
        # DISABLED ON PURPOSE (2026-08-06). `backward` and `offaxis` below are the two reward patches
        # that punish tail-first and sideways travel. They only ever existed because the AMP
        # discriminator was STRUCTURALLY BLIND to travel direction: its motion channels were world-frame
        # magnitudes, so a broadside-sliding fish and a head-first fish produced a bit-identical
        # feature. That is fixed -- the motion block is now the velocity in the body's own
        # (nose,left,up) basis (see AMP_MOTION_NAMES in salmon_amp_tank_env.py) -- so swim DIRECTION is
        # the style reward's job again. Keep BOTH at 0.0 while that is being evaluated: with them on,
        # two mechanisms push the same behaviour and neither result can be attributed.
        backward = 0.0        # SWIM-FORWARD penalty: -w * max(0, -v_forward), i.e. charged ONLY while
                              # the fish is travelling tail-first. `distance` (progress) is
                              # direction-blind -- shrinking the gap pays the same whether the fish
                              # swims forward or reverses -- so with this at 0 a target behind the
                              # fish is most cheaply reached by backing into it, which is what the
                              # fish learned to do. Unlike `heading`, this cannot be farmed by a
                              # motionless fish: at v=0 the penalty is exactly 0.
                              # SIZING (do NOT eyeball this): progress pays distance*dt = 100/30 =
                              # 3.33 per (m/s) of closing speed. With align_gate_floor=0.25 a
                              # REVERSING fish still earns 0.25*3.33 = 0.83 per (m/s). So keep
                              #     backward < 0.83   (rule: < distance*dt*align_gate_floor)
                              # or reversing goes NET NEGATIVE and the fish freezes instead of
                              # turning -- swapping our bug for the worse, already-documented one.
                              # Use 0.3-0.6. At 0.5: forward +0.333/step vs reverse +0.033/step,
                              # so turning is 10x better but reversing still beats doing nothing.
                              # BLIND SPOT: this term is EXACTLY ZERO for sideways motion (v_fwd=0),
                              # which is the failure actually observed -- use `offaxis` instead.
        offaxis = 0.0         # SWIM-HEAD-FIRST penalty: -w * (|v| - max(0, v.heading)).
                              # Charges ANY speed that is not head-first, so it covers BOTH reversing
                              # AND sideways sliding; head-first motion costs exactly 0.
                              # WHY THIS EXISTS: measured 2026-08-06 on the deterministic rollout, the
                              # angle between travel direction and the body line was a median 70 deg,
                              # sideways in 98% of frames -- the fish is sliding broadside, not
                              # swimming. (Confirmed twice: silhouette tracking, and the speed decay
                              # fitting the BROADSIDE drag length 0.044 m instead of head-first 0.835 m.)
                              # `backward` cannot see this at all: sideways means v_fwd = 0, so it
                              # charges nothing. The align gate only trims the pay to the floor.
                              # USE ONE OR THE OTHER: `offaxis` already includes the reversing case,
                              # so set backward=0 when offaxis>0 or tail-first gets charged twice.
                              # SIZING: same ceiling as `backward` -- keep
                              #     offaxis < distance*dt*align_gate_floor = 100/30*0.25 = 0.83
                              # so sideways still beats freezing. Use 0.3-0.6; at 0.5 head-first pays
                              # 0.333/step and sideways 0.033/step, a 10x preference.

    rew_scales: RewardScalesCfg = RewardScalesCfg()
    # ALIGNMENT GATE on the progress/approach GAINS (not the losses).
    # gate = align_gate_floor + (1 - align_gate_floor) * max(0, cos(heading, target_dir))
    # 1.0 = OFF (current behaviour: closing the gap pays the same in any direction).
    # e.g. 0.25 = closing while facing the target pays 4x what closing while facing away pays,
    # which makes "turn first, then swim" the cheapest route to a target behind the fish.
    # Applied ONLY where progress/approach are POSITIVE, so retreating is still punished at full
    # weight (otherwise facing away would become a cheap way to soften the penalty for losing ground).
    # This is NOT the old `heading` term: it multiplies MOVEMENT, so a fish that turns to face the
    # target and then stops earns exactly nothing -- the turn-in-place local optimum is unreachable.
    align_gate_floor = 1.0
    approach_scale = 0.4      # (m) length-scale of the potential-based approach reward (smaller = steeper
                              # near the target). ~ the success radius is a good default.
    spawn_glide_speed = 0.0   # (m/s) initial forward velocity at each reset, along the heading. >0 kills
                              # the cold-start-from-rest tax (fish begins already gliding). ~0.15 is gentle.

    # random / multi-target task (default OFF -> fixed single target). Training-2 turns these on.
    random_target = False     # each reset: sample a NEW target (even bearing + normal distance)
    multi_target = False       # on reaching a target, resample a new one (don't end the episode)
    # HARD ceiling on any sampled target distance (m). Must be >= dist_curriculum_max or the distance
    # curriculum is silently capped here regardless of its own max.
    target_dist_clamp_max = 2.0
    target_dist_mean = 1.0     # normal-distributed target distance: mean (m)
    target_dist_std = 0.3      # normal-distributed target distance: std (m)
    # per-step reward-term printout for env 0 (play/debug). One line per control step listing every
    # reward term's value plus the total (the AMP env appends r_style on the same line). Costs a few
    # GPU syncs per step -- keep OFF for training, turn on at play with env.print_reward_terms=true.
    print_reward_terms = False
    print_reward_every = 1    # print every Nth control step (raise to thin the stream)

    # FRONT-CONE target spawning: 0 = targets appear all around the fish (uniform 360 deg bearing,
    # the historical behaviour). >0 = every target (reset AND multi_target resample) spawns within
    # +/- this many degrees of the fish's CURRENT nose azimuth -- the fish never has to turn more
    # than the cone to face its goal, isolating "swim forward" from "turn around". Distance is
    # untouched (same normal distribution + distance curriculum).
    target_front_cone_deg = 0.0

    # ADAPTIVE CURRICULUM on the success radius: start easy (0.5 m), shrink toward hard (0.2 m) only as
    # the fish PROVES competence (accumulates reaches). This keeps the task reward positive so it isn't
    # drowned by the AMP style reward (the failure mode of the first random-target run).
    curriculum = False
    curriculum_radius_start = 0.5     # success radius at the start (easy)
    curriculum_radius_min = 0.2       # final (hard) success radius
    curriculum_radius_shrink = 0.02   # shrink step per milestone
    curriculum_reaches_per_shrink = 300   # cumulative reaches (all envs) to earn one shrink step
    launch_near_dist = 0.6            # (m) the anti-launch speed penalty ramps in inside this distance
    z_band_halfwidth = 0.3            # (m) free depth band around the target height (see rew_scales.z_band)

    # DISTANCE curriculum (the inverse lever): spawn targets CLOSE first so reaching is the COMMON case
    # (dense success signal + chaining practice grooves the final approach), then GROW the spawn distance
    # as reaches accumulate. Fix the radius (curriculum=False, success_radius=0.4) while this is on --
    # one curriculum at a time keeps attribution clean.
    dist_curriculum = False
    dist_curriculum_start = 0.6       # (m) initial target_dist_mean (near: ~0.2 m past the 0.4 radius)
    dist_curriculum_grow = 0.1        # (m) distance-mean growth per milestone
    dist_curriculum_max = 1.5         # (m) final target_dist_mean
    dist_curriculum_reaches_per_grow = 300   # cumulative reaches (all envs) to earn one growth step
    # RATE-GATED variant (dist_curriculum_gated=True): promote on a good RECENT rate, demote on a bad
    # one. Cures the cumulative-count flaw (a failing policy still scrapes 300 reaches eventually, so
    # difficulty ratchets up regardless of competence -- observed as the 0.26->0.03 reach-rate decay).
    dist_curriculum_gated = False
    dist_curriculum_min = 0.6         # (m) demotion floor
    dist_gate_window = 2000           # control steps between gate decisions (~85 episode-equivalents
                                      # at 256 envs / 6000-step episodes)
    dist_gate_promote = 0.20          # reaches/episode-equiv needed to push the distance OUT
    dist_gate_demote = 0.05           # below this the distance steps back IN (re-groove the skill)

    # target / episode
    target_offset_xy = (1.0, 1.0)   # meters from each env origin; overridden by --target_offset_xy
    target_height = 1.0             # match the fish's swim height (zero gravity)
    target_root_height = 1.0
    success_radius = 0.6   # the deterministic (mean) policy asymptotes to ~0.5975 m from the
                           # target; at radius 0.5 it could NEVER trigger success -> never reset ->
                           # sat-action limit cycle (joint-vel twitch to ~13). 0.6 lets the mean
                           # policy fire success @ ep_step 196 and reset cleanly (max jvel 2.75, no
                           # twitch). Training looked "100% success" at 0.5 only because fixed_sigma
                           # exploration noise jittered position across the 0.5 line; the noiseless
                           # mean cannot. See REPRODUCE_swim_to_target.md.
    termination_height = -5.0
    joint_limit_margin = 0.1
    # blow-up guards
    root_position_limit = 12.0
    joint_velocity_limit = 200.0
    # OPTIONAL FEM soft-body blow-up guard: if set (m/s) and the deformable soft view exists, also
    # flag a blow-up when the max nodal velocity exceeds this. Catches a pure-FEM divergence that
    # hasn't yet propagated to the joints (the joint guard alone would miss it). None = off (default,
    # so existing tasks are unchanged); the coarse-dt stress test sets it.
    fem_velocity_limit = None
    # FEM CFL CIRCUIT-BREAKER (L1): per-substep cap (m/s) on each soft-body nodal-velocity MAGNITUDE.
    # Bounds a CFL divergence WITHOUT a reset (converts exponential growth -> bounded drift), which also
    # kills the deterministic reset-loop at its source. Transparent below the cap (normal nodal vel
    # <~1 m/s -> the PhysX setter is never called on clean steps). None = off (default, so the 1/960
    # swim/IL paths are byte-for-byte unchanged). Needs _soft_view (with_deformable + reset_deformable).
    fem_nodal_vel_clamp = None
    # BLOW-UP COOLDOWN (L2 backstop): control steps to zero-action + neutralize reward/disc after a
    # BLOW-UP reset, so the freshly-reset FEM (whose internal solver state PhysX cannot flush) settles
    # instead of re-diverging at step 1. 0 = off (default).
    blowup_cooldown_steps = 0

    # randomization OFF for now (debug/bootstrap): every reset is identical -> fish always
    # starts at the same place + same heading. Re-open these for robustness once it can swim.
    initial_root_pos_range = (0.0, 0.0)
    initial_root_rot_range = (0.0, 0.0)
    initial_joint_pos_range = (-0.0, 0.0)
    initial_joint_vel_range = (-0.0, 0.0)

    target_marker_height = 1.0
    target_marker_cfg: sim_utils.CuboidCfg = TARGET_MARKER_CFG

    # ---- top-down rolling video of env 0 (one mp4 per PPO epoch = horizon_length steps;
    #      keep only the most recent N on disk). Needs --enable_cameras at launch.
    #      ON by default; the deformable launcher disables it (record + FEM stalls). ----
    record_video = True
    video_dir = str(Path(__file__).resolve().parents[6] / "outputs" / "swim_videos")
    video_clip_steps = 256     # env-steps per clip == one epoch (horizon_length)
    video_keep = 5             # rolling: keep only the most recent N clips on disk
    video_fps = 30
    video_cam_res = 512        # square top-down image (px)
    video_cam_height = 3.0     # camera altitude above the fish (m); frames a ~2.6 m area

    # 3D-trajectory recorder for env0 -- NO camera/RTX needed (just root positions). One PNG
    # (matplotlib 3D) + one Wavefront OBJ per SAVED env0 episode, target marked. Each RUN writes into
    # its OWN timestamped subfolder of traj_dir (run_YYYYmmdd_HHMMSS/), and ALL saved trajectories are
    # KEPT (no rolling deletion -- the old rolling/last-N scheme is removed). To avoid one-file-per-
    # episode clutter, only every traj_save_every_n_episodes-th completed env0 episode is written.
    record_trajectory = True
    traj_dir = str(Path(__file__).resolve().parents[6] / "outputs" / "swim_trajectories")
    traj_save_every_n_episodes = 5   # save env0's path every Nth completed episode (keep ALL)
    # LIVE monitoring: also overwrite a FIXED-name traj_live_env0.{png,obj} every this many control
    # steps MID-episode (in the run's subfolder), so env0's path can be watched building up without
    # waiting for an episode to end. 0 disables. 150 steps ~= one live update every few minutes.
    traj_live_every = 150
