# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the reactive natural-swimming AMP tank task (steps 3 of AMP_TANK_SWIM_PIPELINE.md).

Cooked-FEM zebrafish in a square tank with four collidable walls (from the tank task), trained to
swim + turn like a real zebrafish (env-owned AMP discriminator over the spine-bend/yaw/speed motion
feature vs the carved ZeF reference) while reacting to the walls -- reference-free, forever, with an
energy penalty. No goal/target: the walls are the situation.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .salmon_tank_swim_cfg import SalmonTankSwimEnvCfg

# ZebraFish-05 reference: the SAME individual the fish asset was carved from (single fish, 0% occlusion,
# gate-PASS). Replaces the ZebraFish-01 reference, which was 44% occluded (2 fish) -> carving noise the
# discriminator separated from itself (AUC 0.99) -> r_style pinned flat. See reference_quality_gate.py.
#
# _v2 (2026-08-06) = ORIENTATION-AWARE rebuild, 46 dims (40 bend + 6 motion) instead of 44 (40 + 4).
# The v1 motion block was world-frame/magnitude only, so Phi could not tell head-first from broadside
# from tail-first travel; v2 writes the velocity in the body's own (nose,left,up) basis. The env's
# _amp_feature was changed to match -- the two MUST be rebuilt together, so do not point this back at
# the v1 file without reverting salmon_amp_tank_env.py. Rebuild with:
#     python scripts/benchmark/build_amp_ref_from_fish.py --tag <fish_tag>
# feat_dim is read from this file at runtime (_init_disc), so the disc resizes automatically; an old
# 44-dim disc_state.pth is NOT loadable (disc_load_on_init defaults False, so training is unaffected).
_REF = Path(__file__).resolve().parents[6] / "outputs" / "zef05_amp_ref" / "amp_reference_3d_v2.npz"


@configclass
class SalmonAMPTankEnvCfg(SalmonTankSwimEnvCfg):
    # no goal/target -- reference-free reactive swimming; the walls are the task
    draw_target_marker = False

    # ---- AMP reference motion dataset (spine-bend profile + yaw + speed; incl. turns) ----
    amp_reference_path = str(_REF)
    profile_len = 20                 # K: spine-bend profile length (MUST match extract_amp_features)

    # ---- env-owned discriminator ----
    # ANTI-SATURATION (2026-07-19): the ZebraFish-01 disc saturated (AUC 1.0, r_style pinned) partly on
    # BAD DATA (now fixed: clean ZebraFish-05) and partly on OVER-CAPACITY (76k params vs ~449 real
    # transitions). Smaller net + lower LR + instance noise keep the disc near the GAN equilibrium
    # (D~0, r_style~0.75) instead of winning outright (D<=-1, r_style->0, zero gradient).
    disc_hidden = (128, 128)         # was (256,256): ~500x over-parameterized for this dataset
    # RETRAIN A: R1 does the anti-saturation work (not the update ratio). Ratio dropped to ~2:1 and LR
    # halved so R1 + soft labels + decayed noise keep D near equilibrium with a live gradient. Earlier
    # ratio-only tuning (64:1 saturates, 2:1 too weak, 8:1 middle) was fighting the symptom; R1 is the
    # principled fix (zero-centered gradient penalty on the data manifold, Mescheder 2018).
    disc_lr = 1.5e-5                 # halved from 3e-5
    disc_update_every = 128          # ~2 disc updates per 256-step rollout (≈2:1)
    disc_updates_per_burst = 1
    disc_batch = 512
    disc_r1 = 5.0                    # R1 zero-centered grad penalty on REAL inputs (tune 1-10 by D-spread)
    disc_grad_penalty = 5.0          # legacy alias (disc_r1 takes precedence)
    disc_real_label = 0.9            # SOFT label: real target 0.9 (not 1.0) -> less overconfident D
    # instance noise DECAYED to 0 over disc_noise_decay updates: strong early, gone late.
    disc_noise_std = 0.3
    disc_noise_decay = 2000.0
    # persist disc + optimizer state so eval/deploy reads the TRAINED disc (r_style/AMP-score metric is
    # meaningless with a fresh random disc) and training can warm-restart. Written every N disc updates.
    disc_ckpt_path = str(Path(__file__).resolve().parents[6] / "outputs" / "zef05_amp_ref" / "disc_state.pth")
    disc_ckpt_every = 40             # disc updates between saves (~2/epoch -> every ~20 epochs)
    disc_load_on_init = False        # True to warm-start from disc_ckpt_path (eval sets this)
    disc_buffer_size = 40000         # policy-transition replay buffer

    # ---- reward weights ----
    # AMP-PRIMARY design (2026-07-19): the discriminator (over the clean ZebraFish-05 reference) is the
    # PRIMARY behavioral objective; a small saturating positive-progress term breaks the fwd/back
    # symmetry toward self-directed forward swimming; everything else is minimal numerical safety.
    # The old 0.12 m/s speed TRACKER and the lateral/slip penalties are REMOVED -- AMP is meant to learn
    # the natural forward/lateral distribution, and hand-specifying speed fought the style term.
    # KINEMATIC IMITATION (2026-07-22, remedy b): a phase-clocked TRACKING reward on the bend profile
    # is the primary driver. Unlike AMP/displacement, tracking is ANTI-noise -- exploration noise
    # DEGRADES the match, so the deterministic MEAN must produce the undulation itself (fixes the freeze
    # that survived the disc-fix, displacement, and sigma cut). Target = a traveling wave built from the
    # ZebraFish-05 reference's own envelope + frequency + wavenumber. AMP kept as a small STYLE refiner.
    # obs includes the 2-dim kinematic-imitation phase clock (sin,cos) -> obs=77. Set False to drop it
    # (obs=75), which matches the pre-kinematic July-18 forward-swimmer checkpoints so they deploy here.
    # WARM-START-REFINE run: OFF, so we warm-start FROM the July-18 swimmer (obs=75) and refine its style.
    use_phase_clock = False
    # r_track (kinematic exact-wave tracking) DROPPED: unachievable (the fish can't reproduce ZeF05's
    # wavenumber) and it needs the phase clock (now off). Style comes from the AMP disc; swimming is kept
    # by the high direction-free displacement reward + warm-start from a policy that already swims.
    w_track = 0.0
    track_freq = 1.93                # Hz, reference tail-beat
    track_wavenumber = -8.12         # rad head->tail spatial phase gradient (~1.29 waves on the body)
    track_sharpness = 1.5            # softer: partial tracking rewarded (err 0.6->0.41), guides in
    bend_envelope_path = str(Path(__file__).resolve().parents[6] / "outputs" / "zef05_amp_ref" / "bend_envelope.npy")
    w_style = 0.5                    # AMP style refiner: pulls the warm-started swimmer toward ZeF05 style
                                     # (moderate, so it refines WITHOUT overpowering the swim-keeping disp).

    # REFERENCE STATE INITIALIZATION (RSI) -- the root-cause fix for the deterministic FREEZE.
    # Every episode previously reset to the straight, STILL default pose, so the policy had to DISCOVER
    # the whole cyclic gait from frozen: coherent undulation is a narrow region of joint-space and random
    # exploration from rest yields ~0 net thrust while still paying jvel/action penalties -> the gradient
    # out of "frozen" points DOWNHILL first (an exploration barrier; measured: ep_10 deterministic
    # joint_vel median 0.009 rad/s, 0.028 BL / 20 s -- a statue). RSI resets a FRACTION of envs to a
    # RANDOM PHASE of a real undulating gait (full state: root, joints, FEM nodal pos+vel from
    # scripts/build_scripted_gait_rsi.py) so the policy only has to CONTINUE a gait, not invent one. The
    # seeded joint velocity DECAYS under PD-to-policy-target, so the policy must LEARN to sustain it --
    # standard DeepMimic/AMP RSI, NOT scripting the answer. Keep (1-rsi_frac) rest-starts so the policy
    # also handles standstill.  [[salmon-fem-material-silent-noop]] pattern reused for the FEM state-set.
    rsi_enable = True
    rsi_reference_path = str(Path(__file__).resolve().parents[6] / "outputs" / "jul18_rsi.npz")
    rsi_frac = 0.8                   # fraction of resets seeded mid-gait (rest start from the still pose)
    rsi_gait_freq = 1.5             # Hz of the recorded scripted gait, to align _phase0 to the seed phase

    # EARLY TERMINATION (ET) -- RSI's mandatory companion (DeepMimic). Without it, with 100 s episodes the
    # policy accepts the RSI seed, damps the gait to frozen within ~1 s, then COASTS the rest of the
    # episode collecting the alive bonus + small style reward (the freeze-coast; ep_10 RSI-only was still
    # frozen). ET TERMINATES (value 0, NOT a bootstrapped truncation -> the frozen state is worth 0) an
    # episode once the fish stops undulating for et_freeze_patience consecutive control steps past a
    # post-reset grace window. Then freezing FORFEITS all future reward, so the value function ranks
    # "keep undulating" strictly above "hold still" -- directly dissolving the "frozen policy is OPTIMAL"
    # reward maximum. Activity = mean |joint_vel| on the LIVE lateral DOF (%3==2).
    et_enable = False                # OFF for warm-start-refine: ET-on-velocity was noise-satisfiable
                                     # (inert, et_freeze_frac=0). Anti-freeze here = warm-start from a
                                     # swimmer + HIGH direction-free displacement reward (un-gameable).
    et_freeze_floor = 0.08           # rad/s: live-DOF mean |joint_vel| below this counts as "not swimming"
    et_freeze_patience = 20          # consecutive frozen control steps (~0.67 s) before ET fires
    et_freeze_grace = 30             # steps after reset before ET can fire (let the RSI transient settle)

    # POSITIVE PROGRESS: r_progress = tanh(clamp(v_forward_bl,0)/sat).  v_forward = -root_lin_vel_b[:,0]
    # (body -X is the nose). Saturating (tanh) so more speed is only gently better -- NOT a target to
    # track -- with a real gradient from standstill. sat = median POSITIVE reference forward speed in
    # body-lengths/s (ZebraFish-05: 2.31 BL/s; NOT the 97th-percentile 0.12 m/s the old tracker used).
    # w_progress RAISED to 0.5 (deviation from the spec's 0.1, required for the actual goal). r_style is
    # direction-of-travel AGNOSTIC (body shape only, not COM velocity) and the fish sits in a strong
    # TAIL-FIRST backward local optimum: across 5 runs it drifted at v_fwd ~ -0.02 BL/s even after the
    # progress term was made signed. r_style is ALIGNED with forward (the reference is a head->tail
    # forward swimmer) so a strong progress push does not fight the gait -- it breaks the backward basin,
    # SIGNED-FORWARD velocity reward DISABLED (w_progress=0): it penalized BACKWARD drift, but the fish
    # ("real fish don't necessarily swim straight and forward") and especially the RSI seed gait drift
    # backward at ck=1 -> a signed-forward term fought the seed. The anti-freeze is now the DIRECTION-FREE
    # displacement-magnitude reward below, so a forward bias is neither needed nor wanted here.
    w_progress = 0.0
    # NET-DISPLACEMENT (magnitude) reward: the anti-freeze. Rewards net HORIZONTAL travel (ANY direction)
    # over a ~2 s window (BL/s), which exploration noise CANNOT fake (a jittering-but-stationary mean
    # random-walks to ~0 net displacement) -- only a coherent MEAN gait earns it. Direction-free, so it
    # does not demand forward-straight swimming nor penalize the backward-drifting RSI seed; r_style still
    # owns the gait SHAPE, this only makes "move coherently" outrank "hold still" (the frozen policy was
    # the reward MAXIMUM at 2582). [[hydro-lift-launch-artifact]]: xy-only, so no reward for vertical escape.
    w_displacement = 1.0             # HIGH: the PRIMARY swim-keeping reward for the warm-start-refine run.
                                     # Un-gameable by noise, and the July-18 swimmer already earns it (0.30
                                     # BL/s), so it anchors swimming while AMP (0.5) refines the style.
    displacement_window = 60         # steps (~2 s at 30 Hz control) over which net travel is measured
    displacement_sat = 0.3           # BL/s: tanh saturation ~ the July-18 swimmer's cruise, gradient to go faster
    # ANTI-FREEZE activity reward DISABLED (w_activity=0): the step-to-step |d bend| version was
    # satisfiable by exploration NOISE while the deterministic mean stayed frozen. Replaced by the
    # net-displacement reward above, which noise cannot fake.
    w_activity = 0.0
    activity_target = 0.010
    body_length = 0.5                # m, the auto-skeleton asset's length (for BL/s normalization)
    # SATURATION SPEED must be in the SIM's speed regime, not the reference's. The ZebraFish-05 median
    # is 2.31 BL/s, but that is a 3.4 cm fish at Re~2e3; the 0.5 m sim fish at Re~7e4 swims 6-16x
    # slower in BL/s (old policy peaked ~0.3 BL/s). At sat=2.31 the tanh NEVER saturates for the sim --
    # it degenerates to a weak near-zero-gradient linear term (0.043/BL/s), too weak to pull the policy
    # out of a backward drift (v_fwd stuck at -0.02 for 16 epochs with a HEALTHY disc). At sat=0.4 the
    # term saturates near the sim's achievable cruise (as the spec intends -- "more speed only gently
    # better") and the standstill gradient is ~6x stronger (0.25/BL/s), enough to bias forward while
    # AMP still owns the gait. This is the sim-vs-reference BL/s scale gap, same lesson as the feature.
    progress_saturation_speed = 0.4  # BL/s (sim cruise regime; NOT the 2.31 BL/s reference median)
    # DISABLED for this experiment (kept computed for logging only; weights 0): the fixed-speed tracker
    # and the lateral/slip penalties. AMP learns the natural forward/lateral distribution.
    w_keepmove = 0.0
    keepmove_target_speed = 0.12     # (unused; r_move logged for comparability only)
    move_speed_sigma = 0.06
    w_lateral = 0.0
    lateral_clip = 0.5
    # SLIP stays DISABLED. Re-enabling it (0.5) was COUNTERPRODUCTIVE: the INSTANTANEOUS |v_lat|/|v| is
    # dominated by the lateral RECOIL every undulation cycle produces (slip_ratio pinned ~0.997 even for
    # a good gait), NOT by net crabbing -- so the penalty taxes the gait's own recoil and suppressed
    # r_style. Net-crabbing would need a windowed net-displacement measure; not worth it. r_progress
    # (signed) already rewards net forward and penalizes net backward; the ck=15 hydro supplies the
    # (weak, ~0.03 BL/s) forward thrust. Forward translation is hydro-limited -- the deliverable is the
    # natural GAIT, which is the actual goal ("swim like a real fish"), not a target speed.
    w_slip = 0.0
    slip_min_speed = 0.02
    # STATIC BEND: DISABLED (w_bend=0) -- kept only so pen_bend/bend_ema_abs stay logged as diagnostics.
    # Calibration on the 2026-07-18 rollout showed this instrument is MIS-TARGETED: mean(EMA^2) is
    # 0.00032 for the policy but 0.00109 for the ZeF reference, so a positive w_bend would punish the
    # TARGET gait ~3x harder than the bad one. Cause: the policy's bend oscillates slowly (0.29 Hz,
    # ~3.4 s period) so a 1 s EMA averages it away, while the reference's short segments never settle.
    # The real defect is an OSCILLATION DEFICIT, not a static-bend excess -- see w_slip note and the
    # beat numbers: policy 0.29 Hz / 0.41x amplitude vs reference 2.88 Hz. Do not re-enable without a
    # better statistic (one that separates posture from a slow gait) and a fresh calibration.
    w_bend = 0.0
    bend_ema_alpha = 0.033           # ~1 s time constant at 30 Hz control (1/30)
    # ROOT ACCELERATION: DISABLED (w_accel=0). A tail beat recoils the COM every half-cycle, so a
    # root-acceleration penalty directly taxes the ~2.88 Hz gait we want. Not in the required
    # safety set; kept computed for logging only.
    w_accel = 0.0
    accel_threshold = 5.0
    accel_clip = 20.0

    w_wall = 0.5                     # penalty for wall proximity (open-water: rays read 1.0 -> auto-0)
    wall_margin = 0.15
    w_alive = 0.05                   # per-step alive bonus (small, prevents early-termination gaming)
    # ENERGY PENALTY: DISABLED (0.0). Any positive energy penalty rewards the frozen-pose collapse
    # (still fish = zero power) when forward progress is hydro-limited and the disc has saturated. The
    # anti-freeze activity reward now sets the motion level; AMP shapes it. Re-enable only once forward
    # swimming works and you want cost-of-transport efficiency.
    w_energy = 0.0
    # JOINT-VELOCITY SAFETY: penalize only jvel_rms ABOVE a soft threshold well clear of a healthy gait
    # (~2-4 rad/s) but below the 200 rad/s blow-up limit, so the term is ZERO for the real gait and only
    # discourages approaching divergence. (The old mean(jvel^2) taxed the gait itself; see the smoke test.)
    w_jvel = 0.01
    jvel_soft_threshold = 12.0       # rad/s: pen_jvel = clamp(jvel_rms - this, 0)^2  (0 for a real gait)
    w_action_rate = 0.01             # action smoothness (verified small for a 2.88 Hz gait in smoke)
    # BLOW-UP handling = DISMISSAL, not punishment (decision 2026-07-17): on a divergence step the
    # env's reward is ZEROED and its bend feature is excluded from the disc buffer; the episode
    # terminates (value bootstrap 0 = the naturally-scaled disincentive) and the cooldown masks the
    # post-reset settle. No explicit penalty: at ~0.0002% frequency it is gradient-invisible and a
    # penalty for a numerical artifact risks training conservatism. w_blowup retained for reference
    # only (unused by the dismissal path).
    w_blowup = 0.0

    # CLAMP REMOVED (dt-sweep protocol): the policy must never rely on a velocity clamp at deploy time,
    # so stability is certified by the dt alone — the sweep finds the coarsest dt with zero/rare
    # blow-ups, unassisted. The cooldown stays: it only acts AFTER a blow-up reset (zero-action settle)
    # and is harmless when no blow-ups occur. fem_velocity_limit (=50, autoskel cfg) stays ARMED.
    fem_nodal_vel_clamp = None
    blowup_cooldown_steps = 6

    # wall-distance perception: rays cast in the body-XY plane (forward, left, back, right, + diagonals)
    n_wall_rays = 8
    # OPEN-WATER mode: False removes the tank — walls are not spawned, the wall rays read max-distance
    # (1.0, so the wall penalty auto-zeroes and the policy sees "no wall anywhere"), and bone colliders
    # are pointless. The 6 m root_position_limit becomes the de-facto arena; z containment stays.
    spawn_tank_walls = True

    # ---- VERTICAL containment (the tank is a fence: walls span z 0..2 with open top/bottom, and in
    # zero-g projected_gravity_b is a ZERO vector -- so without these the policy can neither sense nor
    # be discouraged from vertical escape; real thrust exploited that immediately, diving out the
    # bottom at 16 m/s). Obs gains body-frame world-up (attitude) + z-error; reward gains a z-band
    # penalty; termination gains z-out-of-tank and a root-speed sanity guard (treated like a blow-up
    # dismissal: non-physical state, reward zeroed, not punished).
    w_zband = 0.0                    # pen_z REMOVED for 3D: the discriminator now sees up/down speed +
                                     # pitch + vertical bend, so it governs depth/attitude (not a hand rule)
    z_band = 0.4                     # m: no penalty within +/- z_band of target_root_height (=1.0)
    z_min, z_max = 0.0, 2.0          # m: terminate (oob) outside the walls' vertical span
    root_speed_limit = 4.0           # m/s: non-physical for this fish -> dismissed termination.
    never_reset = False              # PLAY-FOREVER viewing: env.never_reset=true -> no timeout/oob/speed
                                     # reset (only a numerical blow-up resets); for GUI watching, NOT training.
                                     # (2.0 clipped healthy episodes: instantaneous root speed spikes
                                     # to ~1.9 from FEM/attachment jitter atop ~0.17 real motion; the
                                     # runaway this guards against was 16 m/s, so 4.0 keeps margin.)

    # long episodes; reset only on failure (blow-up guard). episode_length_s / (decimation*SIM_DT=1/30)
    # = 100.0 * 30 = 3000 control steps.
    episode_length_s = 100.0

    # AMP training likes many parallel envs; FEM is heavy, so a moderate count.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=8.0, replicate_physics=False,
    )

    debug_print = True
