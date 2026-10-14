"""Start a dedicated Duckiematrix, wait for its camera, train, then clean up."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from datetime import datetime
import fcntl
import math
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time

from scripts.find_matrix_offset import MAX_OFFSET, find_free_offset
from training.train import parse_args as validate_training_args


def stop_process(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--learning-rate', type=float, default=.00001)
    parser.add_argument('--speed', type=float, default=.35)
    parser.add_argument('--baseline-model', type=Path, default=Path('baseline.zip'))
    parser.add_argument('--port-offset', type=int, default=11, help='first offset to try; busy sets are skipped')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--train-minutes', type=float, default=25.)
    parser.add_argument('--budget-minutes', type=float, default=30.)
    parser.add_argument('--map', default='loop')
    parser.add_argument('--entity', default='map_0/vehicle_0')
    parser.add_argument('--startup-timeout', type=float, default=120.)
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args(argv)
    if any(option.split('=')[0] in ('--warm-start', '--run-name') for option in extra):
        parser.error('set --baseline-model for warm starts; the run name is automatic')
    if not 0 <= args.port_offset <= MAX_OFFSET or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error('invalid starting port offset or startup timeout')
    baseline = args.baseline_model
    if not baseline.is_absolute():
        baseline = PROJECT_ROOT / ('models' if len(baseline.parts) == 1 else '') / baseline
    if not baseline.is_file() or baseline.suffix != '.zip':
        parser.error(f'Warm-start checkpoint must be an existing .zip file: {baseline}')
    # Validate before allocating ports or launching any processes.
    training_options = ['--learning-rate', str(args.learning_rate), '--speed', str(args.speed),
                        '--steps', str(args.steps), '--train-minutes', str(args.train_minutes),
                        '--budget-minutes', str(args.budget_minutes), '--entity', args.entity,
                        '--warm-start', str(baseline), '--floor-filter', *extra]
    validate_training_args(training_options)
    matrix = ready = training = lock = None
    engine_name = None
    try:
        if not args.dry_run:
            for executable in ('dts', 'docker'):
                if shutil.which(executable) is None:
                    raise RuntimeError(f'Missing command: {executable}')
            lock = open(Path(tempfile.gettempdir()) / f'duckie-matrix-launch-{os.getuid()}.lock', 'a')
            deadline = time.monotonic() + args.startup_timeout
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Timed out waiting for another simulator startup to finish')
                    time.sleep(.2)
        offset = find_free_offset(args.port_offset)
        print(f'Selected port offset {offset} (search started at {args.port_offset}); simulator port {7501 + offset}', flush=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        name = f'dqn_simple_{timestamp}_lr{args.learning_rate:g}_speed{args.speed:g}_offset{offset}'
        engine_name = f'dqn-matrix-{name}'
        matrix_cmd = ['dts', 'matrix', 'run', '--standalone', '--embedded', '--map', args.map,
                      '--no-tutorial', '--static-ports', '--port-offset', str(offset),
                      '--engine-name', engine_name, '--local']
        train_cmd = [sys.executable, '-u', str(PROJECT_ROOT / 'training/train.py'),
                     *training_options, '--port-offset', str(offset), '--run-name', name]
        print('Simulator command:', shlex.join(matrix_cmd), flush=True)
        print('Training command:', shlex.join(train_cmd), flush=True)
        print(f'Model: models/{name}_final_<actual_steps>_steps.zip', flush=True)
        if args.dry_run:
            return 0
        logs = PROJECT_ROOT / 'training/logs'
        logs.mkdir(parents=True, exist_ok=True)
        matrix_log = logs / f'{name}_matrix.log'
        with matrix_log.open('w') as stream:
            matrix = subprocess.Popen(matrix_cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Waiting for camera; simulator log: {matrix_log}', flush=True)
        ready = subprocess.Popen([sys.executable, '-u', str(PROJECT_ROOT / 'scripts/wait_matrix.py'),
                                  '--entity', args.entity, '--port', str(7501 + offset),
                                  '--timeout', str(args.startup_timeout)], start_new_session=True)
        deadline = time.monotonic() + args.startup_timeout
        while ready.poll() is None:
            if matrix.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(f'Simulator startup failed or timed out. See {matrix_log}')
            time.sleep(.2)
        if ready.returncode:
            raise RuntimeError(f'Camera startup failed. See {matrix_log}')
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        lock = None
        training = subprocess.Popen(train_cmd, start_new_session=True)
        code = training.wait()
        print(f'Run summary: {logs / name / "summary.json"}', flush=True)
        return code if code >= 0 else 128 - code
    except KeyboardInterrupt:
        print('Stopping this run and saving its current checkpoint.', flush=True)
        return 130
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        for process in (training, ready, matrix):
            stop_process(process)
        if matrix is not None:
            try:
                # Match docker.from_env() in dts, including its default socket.
                subprocess.run(['docker', '--host', os.environ.get('DOCKER_HOST', 'unix:///var/run/docker.sock'),
                                'stop', '--time', '5', engine_name], timeout=15,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(main())
