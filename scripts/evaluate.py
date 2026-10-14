"""Deterministic driving evaluation with failure, alignment, and speed metrics."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from collections import Counter
import json
import math
import time

from stable_baselines3 import DQN

from training.environment import ACTION_NAMES, DiscreteSteering, RoadCamera, FrameStackCamera
from training.reward import LineReward
from utils.utils import saved_speed, saved_steering_gain, validate_model


def evaluate_run(model, env, duration, *, start_action=2, offset_steps=0, deadline=None):
    """Count failures separately from time limits, including initial perturbations."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Evaluation duration must be positive and finite")
    started = time.monotonic()
    stop_at = min(started + duration, deadline) if deadline is not None else started + duration
    total_reward = 0.
    steps = failures = time_limits = both_lines = 0
    errors = []
    actions = Counter()
    reasons = Counter()
    longest_streak = streak = 0
    try:
        observation, _ = env.reset()
        while time.monotonic() < stop_at:
            if steps < offset_steps:
                action = start_action
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            actions[int(action)] += 1
            both_lines += info.get("detection_state") == "both"
            if info.get("lane_center_error") is not None:
                errors.append(abs(info["lane_center_error"]))
            if terminated:
                failures += 1
                reasons[info.get("termination_reason") or "unknown"] += 1
                streak = 0
            else:
                streak += 1
                longest_streak = max(longest_streak, streak)
            if truncated:
                time_limits += 1
            if terminated or truncated:
                if time.monotonic() < stop_at:
                    observation, _ = env.reset()
    finally:
        env.unwrapped.robot.motors.set_pwm(left=0., right=0.)
    elapsed = time.monotonic() - started
    mean_error = sum(errors) / len(errors) if errors else None
    completed = elapsed >= duration * .98 and steps > 0
    # A camera-based screening test, not proof of completed laps.
    passed = (completed and failures == 0 and both_lines / max(steps, 1) >= .90
              and mean_error is not None and mean_error <= .30
              and steps / max(elapsed, 1e-9) >= 10.)
    result = dict(seconds=elapsed, requested_seconds=duration, steps=steps,
                  fps=steps / max(elapsed, 1e-9), failures=failures, time_limits=time_limits,
                  reward_total=total_reward, reward_per_step=total_reward / max(steps, 1),
                  both_lines_fraction=both_lines / max(steps, 1),
                  mean_abs_center_error=mean_error, longest_clean_streak=longest_streak,
                  actions={name: actions[i] for i, name in enumerate(ACTION_NAMES)},
                  termination_reasons=dict(reasons), completed=completed, passed=passed,
                  start_action=ACTION_NAMES[start_action], offset_steps=offset_steps)
    print(f"evaluation steps={steps} fps={result['fps']:.1f} failures={failures} "
          f"reward/step={result['reward_per_step']:.3f} both_lines={result['both_lines_fraction']:.0%} "
          f"passed={passed}", flush=True)
    return result


def evaluate_runs(model, env, runs=5, duration=60., *, deadline=None, vary_starts=True):
    results = []
    for run in range(runs):
        if deadline is not None and time.monotonic() >= deadline:
            break
        action = (2, 1, 3, 0, 4)[run % 5] if vary_starts else 2
        print(f"Evaluation run {run + 1}/{runs}, start={ACTION_NAMES[action]}", flush=True)
        results.append(evaluate_run(model, env, duration, start_action=action,
                                    offset_steps=5 if action != 2 else 0, deadline=deadline))
    return dict(runs=results, requested_runs=runs,
                passed=len(results) == runs and all(r["passed"] for r in results),
                failures=sum(r["failures"] for r in results),
                note="Camera-based lane-following check; laps and physical-bot transfer are not measured.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="saved zip; defaults to newest in models/")
    parser.add_argument("--runs", "--episodes", dest="runs", type=int, default=5)
    parser.add_argument("--duration", type=float, default=60.)
    parser.add_argument("--entity", default="map_0/vehicle_0")
    parser.add_argument("--port", type=int, default=7501, help="simulator port: 7501 + offset")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--same-start", action="store_true", help="disable initial steering perturbations")
    parser.add_argument("--output", type=Path, help="write metrics as JSON")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.runs <= 0 or not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("runs and duration must be positive and finite")
    path = args.model
    if path is None:
        paths = list((PROJECT_ROOT / "models").glob("*.zip"))
        if not paths:
            parser.error("No saved models found")
        path = max(paths, key=lambda p: p.stat().st_mtime_ns)
    print(f"Evaluating model: {path}", flush=True)
    model = DQN.load(path, device=args.device)
    validate_model(model)
    reward_env = LineReward(entity_name=args.entity, camera_settings=model.camera_settings, port=args.port)
    for name, value in getattr(model, "reward_settings", {}).items():
        if name in reward_env.reward_settings():
            setattr(reward_env, name, value)
    env = DiscreteSteering(FrameStackCamera(RoadCamera(reward_env)), speed=saved_speed(model),
                           steering_gain=saved_steering_gain(model))
    report = dict(model=str(path), reward_version=reward_env.reward_version, passed=False, runs=[])
    try:
        report.update(evaluate_runs(model, env, args.runs, args.duration, vary_starts=not args.same_start))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            env.close()
        finally:
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
