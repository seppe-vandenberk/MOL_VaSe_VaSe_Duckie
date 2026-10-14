# MOL_VaSe_VaSe_Duckie

The CNN receives 32×32 binary line masks. `FRAME_COUNT` in
`training/environment.py` controls history length (currently one frame).
The lower half of each RGB camera frame is resized, then HSV thresholds select
white and yellow paint. Both colors become white (255); all other pixels become
black (0). SB3 normalizes the masks to 0 and 1. Color thresholds are stored in
each new model's camera settings.

Training and real-bot previews show the raw camera, the RGB crop with line
detections, and the exact masks supplied to the CNN. Press Q to stop.
Train a new policy with `python training/train.py`; older RGB checkpoints are
incompatible with the binary input. Older image sizes also need
retraining for the new 32×32 input. A small CNN handles this reduced resolution.
Preview a real camera without driving using
`python scripts/view_real.py BOT_NAME`.

The crop keeps the bottom 50% and the full camera width. Resizing uses area
interpolation. It preserves the camera's native lens distortion; it does not
widen its optical field of view. Duckiematrix's current loop camera already has
calibrated lens distortion enabled.

Training uses at most 30,000 steps or 25 minutes, whichever comes first, then
five deterministic 60-second evaluation runs within a 30-minute overall budget.
Camera waiting, resets, and learning count toward the budget. Saving and component
shutdown may finish just after the deadline. The preview is off by default;
use `--preview` to enable it. Duckiematrix must be running and producing frames.

```bash
python training/train.py
# Short run for inspecting throughput and failures before a full experiment:
python training/train.py --steps 2000 --train-minutes 2 --budget-minutes 3 --eval-runs 1 --eval-duration 30
# Initialise weights from a compatible checkpoint; replay and optimiser start fresh:
python training/train.py --warm-start models/YOUR_MODEL.zip
python scripts/evaluate.py --model models/YOUR_MODEL.zip --output training/logs/evaluation.json
```

The short-run DQN schedule starts learning after 500 steps, trains once per
environment step with batches of 64, updates its target every 500 steps, and
reduces exploration over the first 20% of the requested steps. The replay buffer
holds 30,000 transitions. `config.yaml` still controls speed and learning rate.
`--device cpu`, `--device cuda`, and `--seed` allow repeatable timing comparisons.

Each run has its own folder under `training/logs/`, with episode rewards,
256-step reward batches, `progress.csv`, and `summary.json`. Progress records
actual steps/second, failures, time-limit resets, boundary visibility, centering
error, action counts, exploration, and learning updates. Existing logs are kept.
Checkpoints retain the camera settings, steering gain, reward version, and full
training settings. A warm start preserves the source speed and steering mapping.

Brief single-boundary loss uses the most recent valid lane width to provide
recovery feedback. Yellow/white paint identifies the left/right boundary even
when it crosses the image centre. Failure occurs after 10 consecutive missing-boundary steps
or 40 steps without recovering both boundaries. A 1,200-step episode cap is a
time-limit truncation, not a driving failure. Steering gain `.6` preserves older
checkpoints' 2.125:1 maximum wheel ratio; `--steering-gain 1` trains with a 4:1
ratio if tighter turns are needed. Deployment uses the saved gain automatically.

Evaluation disables random exploration and varies initial steering slightly
between runs. A run passes the camera-based screening when it completes its
duration with zero failures, at least 90% two-boundary visibility, mean absolute
normalized center error at most 0.30, and at least 10 steps/second. JSON includes
the individual metrics even when screening fails. This does not measure completed
laps, progress along the track, or transfer to a physical robot. Inspect complete
laps before treating a checkpoint as ready for deployment. Evaluation exits with
status 1 when screening fails. Use `--same-start` for controlled comparisons.

Offline checks (no simulator or physical robot required):

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python -m unittest discover -s tests -v
```
