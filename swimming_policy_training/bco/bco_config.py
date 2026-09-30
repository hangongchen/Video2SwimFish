"""All BCO hyperparameters, in one dataclass. Nothing is hardcoded elsewhere in bco/ --
every tunable value used by replay_buffer.py / idm.py / pseudo_labeler.py / bc_trainer.py /
run_bco.py is read from a BCOConfig instance passed in explicitly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BCOConfig:
    """Hyperparameters for every BCO stage. Defaults match the task's spec; override via a
    --config YAML/JSON file (see load_bco_config) and/or individual --flag overrides in
    run_bco.py's CLI."""

    # ---- shared: env + device + bookkeeping -------------------------------------------------
    # SAME env/obs/action-shape/reward/episode-length as the BLM+RL (all-PC PCA) baseline --
    # only the controller differs (raw joint targets instead of PCA coefficients). See
    # source/FISH/FISH/tasks/direct/fish/salmon_swim_reach10_env.py:SalmonSwimJointReach10Env.
    task_id: str = "Template-Salmon-Reach10-Joint-Misty-Direct-v0"
    device: str = "cuda:0"
    seed: int = 42
    checkpoint_dir: str = str(_REPO_ROOT / "checkpoints" / "bco")
    dry_run: bool = False

    # ---- Step 1: random exploration data collection ----------------------------------------
    num_envs: int = 256
    num_random_steps: int = 200_000
    replay_buffer_file: str = "replay_buffer.pt"

    # ---- Step 2: inverse dynamics model (IDM) -----------------------------------------------
    idm_hidden_layers: list[int] = field(default_factory=lambda: [512, 512, 256])
    idm_activation: str = "elu"
    idm_lr: float = 3e-4
    idm_epochs: int = 50
    idm_batch_size: int = 4096
    idm_checkpoint_file: str = "idm.pt"

    # ---- Step 3: pseudo-labeling of expert (video) curvature sequences ---------------------
    expert_curvatures: str = ""           # path to expert_curvatures.npy, shape (T, N)
    pseudo_labeled_file: str = "pseudo_labeled.npz"
    # pseudo-actions are clamped to the env's own action bounds [-pseudo_action_clip,
    # +pseudo_action_clip] before saving -- the IDM is trained on IN-DISTRIBUTION random-policy
    # actions (which lie in [-1,1]); applying it to out-of-distribution expert observations can
    # occasionally overshoot, and downstream BC/PPO both expect actions in [-1,1].
    pseudo_action_clip: float = 1.0

    # ---- Step 4: behavioral cloning (BC) pre-training ---------------------------------------
    bc_epochs: int = 100
    bc_lr: float = 1e-3
    bc_batch_size: int = 256
    bc_val_split: float = 0.1
    bc_policy_file: str = "bc_policy.pth"
    # the SAME rl_games agent yaml the PPO fine-tune stage (and the live BLM+RL baseline) uses,
    # so the BC-pretrained network is architecture-identical and loads via rl_games --checkpoint
    # with zero format mismatch.
    rl_games_agent_yaml: str = str(
        _REPO_ROOT / "source/FISH/FISH/tasks/direct/fish/agents/rl_games_ppo_reach10_joint_cfg.yaml"
    )

    # ---- Step 5: PPO fine-tuning (existing rl_games trainer; NOT reimplemented here) --------
    train_ppo_script: str = str(_REPO_ROOT / "scripts/rl_games/train_ppo.py")
    ppo_num_envs: int = 256
    ppo_max_iterations: int | None = None       # None -> yaml's own max_epochs
    ppo_headless: bool = True
    ppo_extra_args: list[str] = field(default_factory=list)   # verbatim extra CLI args/overrides
    # wandb ("always launch trainings tracked online" -- project convention)
    wandb_track: bool = True
    wandb_project_name: str = "fish_articulation_analytic_water"
    wandb_entity: str | None = None      # None -> your default wandb entity ($WANDB_ENTITY)
    wandb_name: str = "fish_bco_ppo_finetune"
    wandb_key_file: str = "~/.wandb_key"    # or export WANDB_API_KEY

    def resolve(self, filename: str) -> str:
        """checkpoint_dir-relative path helper: an absolute path, OR a relative path that
        already has directory components (e.g. "./checkpoints/bco/bc_policy.pt" -- exactly
        the shared-artifact style the task's own example commands pass for --bc_checkpoint),
        is returned AS-IS relative to the current working directory. Only a BARE filename with
        no directory component (e.g. "bc_policy.pth", the default) resolves under
        self.checkpoint_dir. This lets one shared bc_policy.pt/idm.pt live in a parent
        directory while --checkpoint_dir points each PPO fine-tune run at its own subdirectory."""
        p = Path(filename)
        if p.is_absolute() or len(p.parts) > 1:
            return str(p)
        return str(Path(self.checkpoint_dir) / filename)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}


def load_bco_config(config_path: str | None, overrides: dict[str, Any] | None = None) -> BCOConfig:
    """Build a BCOConfig: defaults, then a --config YAML/JSON file (if given), then explicit
    per-flag `overrides` (only keys with a non-None value are applied, so an unset CLI flag
    never clobbers a value from the config file)."""
    cfg = BCOConfig()
    valid = BCOConfig.field_names()
    if config_path:
        with open(config_path) as f:
            text = f.read()
        file_dict = json.loads(text) if config_path.endswith(".json") else yaml.safe_load(text)
        for k, v in (file_dict or {}).items():
            assert k in valid, f"unknown BCOConfig field {k!r} in {config_path}"
            setattr(cfg, k, v)
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        assert k in valid, f"unknown BCOConfig field {k!r} in CLI overrides"
        setattr(cfg, k, v)
    return cfg
