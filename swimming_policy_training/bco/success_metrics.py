"""THE canonical, reusable success-rate computation for the Reach10 task family.

Background (see project memory / this conversation's history for the full story): the
Reach10 protocol (salmon_swim_reach10_env.py:TenTargetMixin) exposes AT LEAST THREE distinct
per-step scalars that all sound like "success rate" but are NOT interchangeable:

  - `reach10/succ_frac_running` (surfaced by rl_games' FishWandbObserver as `core/success_rate`
    -- **THE WRONG ONE, now deleted from that observer's curated payload**): a LIVE,
    IN-PROGRESS running ratio -- successes-so-far / attempts-so-far within each env's CURRENT,
    possibly-incomplete 10-attempt episode, averaged across whichever envs happen to be live
    at that instant. Systematically biased low relative to the true per-episode rate (envs
    early in a fresh episode contribute a noisy, low-attempt-count ratio), and was mistakenly
    used as "the" success rate earlier in this project -- verified empirically to read ~2x
    LOWER than the correct metric on a near-zero-success run.
  - `reach10/targets_per_ep_deque_LEGACY`: an EARLIER, ALSO-WRONG per-completed-episode
    accumulator with a documented completion-order/capacity-cap bias (a capped deque
    disproportionately drops episodes that end FIRST, i.e. failures/blow-ups, inflating the
    reading). Superseded; kept only as a historical comparison field, never surfaced here.
  - `reach10/targets_per_ep` (surfaced as `core/targets_per_ep`) -- **THE CORRECT ONE**: an
    unbiased EMA over EVERY episode that actually COMPLETED (equal weight per episode, no
    capacity cap, no completion-order selection bias). This is the ONLY function in this
    codebase that should be called "the success rate".

This module is used from TWO different vantage points that cannot share one code path (they
see different data), but BOTH are anchored to the exact same tag name (`SUCCESS_TAG`) so
there is only ONE definition of "which field is correct":
  - `extract_live_success_count`: called INSIDE the training process itself (rl_games'
    FishWandbObserver), operating on its already-built per-epoch `seen` dict.
  - `read_success_series_from_tensorboard`: called from a SEPARATE, later process (the
    dashboard), which has no access to that dict and can only re-read what got logged to the
    run's own TensorBoard event file -- which is fine, because SUCCESS_TAG is written there
    unconditionally by rl_games regardless of what any script's stdout prints.
"""

from __future__ import annotations

import re
from pathlib import Path

# The ONLY env-side field this module will ever read as "success rate". Never
# 'reach10/succ_frac_running' (wrong: live/in-progress) and never
# 'reach10/targets_per_ep_deque_LEGACY' (wrong: completion-order/capacity-cap biased).
SUCCESS_FIELD = "reach10/targets_per_ep"
# The TensorBoard tag rl_games' FishWandbObserver writes SUCCESS_FIELD's value under
# (see scripts/rl_games/train_ppo.py: "Core/" + k.split("/", 1)[1] for every "core/*" payload
# key -- this module's train_ppo.py caller is responsible for actually putting SUCCESS_FIELD's
# value into payload["core/targets_per_ep"], which is what makes this tag exist at all).
SUCCESS_TAG = "Core/targets_per_ep"


def extract_live_success_count(seen: dict) -> float | None:
    """Given the observer's per-epoch `seen` dict (built from env extras -- see
    FishWandbObserver.after_print_stats), return the CORRECT, debiased success count: the
    average number of targets reached per COMPLETED episode (0..n_targets_per_episode, e.g.
    0..10) -- NOT normalized to 0..1 here (the caller may not know n_targets_per_episode;
    dashboard.py's read path normalizes separately using each run's own saved config).
    Returns None if the field is absent (e.g. a non-Reach10 task)."""
    return seen.get(SUCCESS_FIELD)


def read_success_series_from_tensorboard(run_dir: str) -> list[tuple[float, float]]:
    """[(epoch, raw_success_count)] for a COMPLETED-elsewhere training run, read directly from
    its own TensorBoard event file under SUCCESS_TAG (see module docstring for why this is the
    only trustworthy source for a run this process didn't itself train). Empty list if the
    run directory / tag doesn't exist (never raises -- callers render an empty-but-valid panel
    instead of crashing)."""
    if not run_dir:
        return []
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []
    summaries_dir = Path(run_dir) / "summaries"
    if not summaries_dir.exists():
        return []
    ea = EventAccumulator(str(summaries_dir))
    ea.Reload()
    if SUCCESS_TAG not in ea.Tags().get("scalars", []):
        return []
    return [(float(e.step), float(e.value)) for e in ea.Scalars(SUCCESS_TAG)]


def read_n_targets_per_episode(run_dir: str | None, default: int = 10) -> int:
    """cfg.n_targets_per_episode as saved in a run's own params/env.yaml, for normalizing
    read_success_series_from_tensorboard's raw 0..N count to a 0..1 rate. Falls back to
    `default` (the value every Reach10 task in this project has used) if unreadable or if
    run_dir itself is None/empty (an omitted optional arm, e.g. a 2-arm-only dashboard)."""
    if not run_dir:
        return default
    env_yaml = Path(run_dir) / "params" / "env.yaml"
    if not env_yaml.exists():
        return default
    m = re.search(r"^n_targets_per_episode:\s*(\d+)\s*$", env_yaml.read_text(errors="replace"),
                  re.MULTILINE)
    return int(m.group(1)) if m else default


def read_success_rate_from_tensorboard(run_dir: str) -> list[tuple[float, float]]:
    """[(epoch, success_rate)] normalized to 0..1 -- the one-call convenience most callers
    (e.g. the dashboard) actually want: read_success_series_from_tensorboard, divided by
    read_n_targets_per_episode."""
    n = read_n_targets_per_episode(run_dir)
    return [(epoch, count / n) for epoch, count in read_success_series_from_tensorboard(run_dir)]
