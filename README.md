# MOL_VaSe_VaSe_Duckie

Train, evaluate, and test a Duckiebot lane-following DQN. Activate the
`duckie-rl311` environment first. `environment.yml` records its dependencies;
Duckietown Shell (`dts`) and Docker must also be installed for simulation.

The CNN receives a 32×32 binary mask of white and yellow tape from the lower
half of the RGB camera image. Road and background are black. The optional floor
filter rejects purple carpet and broad bright floor patches beside black mats.
The viewers show raw RGB, detected boundaries, and the exact CNN mask.

`models/baseline.zip` is the retained 23,177-step checkpoint previously used for
warm starts. `models/baseline.json` records its origin, checksum, and original
evaluation results. New training runs save separate checkpoints.

Edit the main controls at the top of `scripts/train_matrix.sh`:

```bash
LEARNING_RATE="${LEARNING_RATE:-0.00001}"
SPEED="${SPEED:-0.35}"
BASELINE_MODEL="${BASELINE_MODEL:-baseline.zip}"
STEPS="${STEPS:-30000}"
```

Run the launcher, or inspect its commands first:

```bash
bash scripts/train_matrix.sh --dry-run
bash scripts/train_matrix.sh
# Temporary overrides; BASELINE_MODEL can be a filename under models/ or a path:
LEARNING_RATE=0.00002 SPEED=0.4 BASELINE_MODEL=baseline.zip bash scripts/train_matrix.sh --no-preview
```

The launcher warm-starts from the selected model with fresh replay and optimizer
state, and enables floor filtering. It finds the first complete free set of all
13 simulator ports starting at `PORT_OFFSET` (default 11), waits for the camera,
and starts training. The selected offset appears in both commands and the model
filename. Simultaneous launchers coordinate startup. Stopping the launcher stops
its own simulator and saves available training progress.

Training defaults to 30,000 requested steps or 25 minutes, followed by five
60-second simulator checks within a 30-minute total budget. The preview is on by
default; Q stops training and saves progress. Logs include rewards, progress, and
`summary.json` under `training/logs/`. Checkpoint names include learning rate,
speed, port offset, and actual completed steps. For warm starts, that last count
is the steps added during this run; `warm_start_steps` records the source count.

Inspect a running simulator without moving or resetting its robot. Set
`SIMULATOR_PORT` at the top of `scripts/view_sim.py`, or pass the port printed by
the launcher (`7501 + offset`):

```bash
python scripts/view_sim.py --port 7512
python scripts/view_sim.py --port 7512 --model models/baseline.zip
```

Inspect the real camera without driving. Set `CAMERA_PORT` at the top of
`scripts/view_real.py` if needed (default 11911):

```bash
python scripts/view_real.py ducksolo.local --floor-filter
python scripts/view_real.py ducksolo.local --port 11911 --floor-filter
```

Both camera viewers accept `--model`, `--floor-filter`, `--no-floor-filter`, and
`--road-value-max`. The simulator viewer enables the floor filter by default to
match the launcher. Q closes a camera-only viewer.

Evaluate a saved policy in a running simulator. Without `--model`, evaluation
selects the newest checkpoint in `models/`, including the baseline:

```bash
python scripts/evaluate.py --model models/baseline.zip --port 7512 --output training/logs/evaluation.json
```

Evaluation disables exploration and varies initial steering. A run passes its
camera-based screening with no lane-loss failures, at least 90% visibility of
both boundaries, mean absolute center error at most 0.30, and at least 10
steps/second. These metrics do not establish completed laps or physical transfer.

Run a bounded physical trial with the driving viewer and saved diagnostics:

```bash
python scripts/run_real.py models/baseline.zip ducksolo.local --duration 10 --floor-filter --output training/logs/real_trial
```

Real-camera and steering defaults live in `utils/real_settings.py`:
`REAL_CROP_TOP=0.4` retains the lower 60% of the image, and
`REAL_STEERING_GAIN=1.0` gives a 4:1 outer/inner wheel ratio on full turns.
The real viewer uses the same crop as the driver. Adjust them per run with
`--crop-top` and `--steering-gain`:

```bash
python scripts/view_real.py ducksolo.local --floor-filter --crop-top 0.4
python scripts/run_real.py models/baseline.zip ducksolo.local --duration 10 --floor-filter --crop-top 0.4 --steering-gain 1.0
```

Stronger steering reallocates the same total PWM between wheels; it does not
raise straight-line power. The mask remains 32×32, though changing the crop changes
where markings appear in the policy input. Simulator training retains its saved
camera crop and steering mapping. Physical traction must be checked on the mat.
Trial summaries record the applied crop, steering gain, and individual wheel PWM.

The physical CLI requires both boundaries before moving, allows five seconds
of single-boundary recovery, and stops after one second without either line.
Override these with `--single-boundary-timeout` and `--no-boundary-timeout`.
The preview shows which boundary is missing and the timer. Q stops the motors;
shutdown always sends zero PWM. `MOTOR_GAIN` at the top of `run_real.py` controls
real motor compensation. Explicit `--speed` bypasses that gain.

Run offline regression tests without a simulator or robot:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python -m unittest discover -s tests -q
```

Open `scripts/model_comparison.ipynb` and run all cells to compare every model in
`training/logs`. It plots training reward per step, the fraction of frames showing
both boundaries, weighted training/evaluation averages, and evaluation step counts
for fully completed results. Incomplete runs are excluded from the evaluation step
chart; completed runs that failed screening remain marked as failures. Standalone
`evaluate.py --output` reports saved under `training/logs` are included too.
