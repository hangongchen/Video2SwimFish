"""Behavioral Cloning from Observation (BCO) baseline for the FISH locomotion project.

Pipeline (see run_bco.py):
    1-2. random-policy rollout in the SAME Isaac Lab env the BLM+RL baseline uses (only the
         action space differs: raw joint targets instead of PCA coefficients) -> train an
         inverse dynamics model (IDM) on the collected (s_t, a_t, s_{t+1}) transitions.
    3.   apply the IDM to real-fish expert curvature sequences (video-derived, no action
         labels) to produce pseudo-action labels.
    4.   behavioral-clone an rl_games-compatible policy on the pseudo-labeled data.
    5.   fine-tune that policy with the EXISTING rl_games PPO trainer (scripts/rl_games/
         train_ppo.py), initialized from the BC checkpoint.
"""
