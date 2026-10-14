"""Exercise the Bash launcher with local fake simulator and Docker commands."""

import inspect
import json
import os
from pathlib import Path
import shutil
import socket
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
import numpy as np

from training.train import parse_args
from scripts.wait_matrix import wait_for_camera
from scripts.find_matrix_offset import BASE_PORTS, find_free_offset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MatrixLauncherTests(unittest.TestCase):
    def prepare(self, root):
        (root / "scripts").mkdir()
        (root / "training").mkdir()
        (root / "training/__init__.py").touch()
        (root / "models").mkdir()
        checkpoint = root / "models/baseline.zip"
        checkpoint.touch()
        shutil.copy(PROJECT_ROOT / "scripts/train_matrix.sh", root / "scripts/train_matrix.sh")
        shutil.copy(PROJECT_ROOT / "scripts/find_matrix_offset.py", root / "scripts/find_matrix_offset.py")
        shutil.copy(PROJECT_ROOT / "scripts/launch_training.py", root / "scripts/launch_training.py")
        (root / "training/train.py").write_text(
            "import argparse, math, re, sys, os, json\nfrom pathlib import Path\nTOTAL_TIMESTEPS=30000\n" +
            inspect.getsource(parse_args) + '''
if __name__ == '__main__':
    args=parse_args()
    with open(os.environ['TRACE'], 'a') as stream:
        stream.write(json.dumps(dict(kind='train', args=vars(args)), default=str)+'\\n')
    raise SystemExit(int(os.environ.get('TRAIN_EXIT', '0')))
''')
        (root / "scripts/wait_matrix.py").write_text('''
import sys,os,json
with open(os.environ['TRACE'],'a') as stream:
    stream.write(json.dumps(dict(kind='ready', args=sys.argv[1:]))+'\\n')
''')
        binaries = root / "bin"; binaries.mkdir()
        for name, source in {
            'dts': '''
import sys,os,json,signal,time
with open(os.environ['TRACE'],'a') as stream:
    stream.write(json.dumps(dict(kind='matrix', args=sys.argv[1:]))+'\\n')
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
while True: time.sleep(.1)
''',
            'docker': '''
import sys,os,json
with open(os.environ['TRACE'],'a') as stream:
    stream.write(json.dumps(dict(kind='docker', args=sys.argv[1:]))+'\\n')
''',
        }.items():
            target = binaries / name
            target.write_text(f'#!{sys.executable}\n' + source)
            target.chmod(0o755)
        return dict(os.environ, PATH=str(binaries)+os.pathsep+os.environ['PATH'],
                    PYTHON_BIN=sys.executable, TRACE=str(root/'trace.jsonl'),
                    LEARNING_RATE='0.0002', SPEED='0.4', PORT_OFFSET='2000', STEPS='7', BASELINE_MODEL=str(checkpoint))

    def test_launch_routes_ports_and_cleans_up_on_success_and_failure(self):
        for code in (0, 7):
            with self.subTest(code=code), tempfile.TemporaryDirectory(prefix='matrix launcher ') as folder:
                root=Path(folder); env=self.prepare(root); env['TRAIN_EXIT']=str(code)
                result=subprocess.run(['bash', str(root/'scripts/train_matrix.sh'), '--no-preview'],
                                      env=env, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, code, result.stderr + result.stdout)
                records=[json.loads(line) for line in (root/'trace.jsonl').read_text().splitlines()]
                by_kind={row['kind']:row['args'] for row in records}
                self.assertEqual(set(by_kind), {'matrix','ready','train','docker'})
                self.assertEqual(by_kind['train']['port_offset'], 2000)
                self.assertEqual(by_kind['train']['speed'], .4)
                self.assertEqual(by_kind['train']['learning_rate'], .0002)
                self.assertEqual(by_kind['train']['steps'], 7)
                self.assertEqual(by_kind['train']['warm_start'], env['BASELINE_MODEL'])
                self.assertTrue(by_kind['train']['floor_filter'])
                self.assertIn('9501', by_kind['ready'])
                self.assertIn('_lr0.0002_speed0.4_offset2000', by_kind['train']['run_name'])
                matrix=by_kind['matrix']
                self.assertEqual(matrix[matrix.index('--port-offset')+1], '2000')
                self.assertEqual(by_kind['docker'][-1], matrix[matrix.index('--engine-name')+1])
                self.assertEqual(by_kind['docker'][:2], ['--host', env.get('DOCKER_HOST', 'unix:///var/run/docker.sock')])

    def test_dry_run_starts_nothing_and_managed_overrides_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); env=self.prepare(root)
            result=subprocess.run(['bash', str(root/'scripts/train_matrix.sh'), '--dry-run'],
                                  env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('_final_<actual_steps>_steps.zip', result.stdout)
            self.assertFalse((root/'trace.jsonl').exists())
            result=subprocess.run(['bash', str(root/'scripts/train_matrix.sh'), '--warm-start=bad.zip'],
                                  env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((root/'trace.jsonl').exists())

    def test_default_warm_start_and_learning_rate_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); env=self.prepare(root)
            env.pop('LEARNING_RATE'); env.pop('SPEED'); env.pop('BASELINE_MODEL')
            result=subprocess.run(['bash', str(root/'scripts/train_matrix.sh'), '--dry-run'],
                                  env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--learning-rate 1e-05', result.stdout)
            self.assertIn('--speed 0.35', result.stdout)
            self.assertIn(str(root / 'models/baseline.zip'), result.stdout)
            env['BASELINE_MODEL']=str(root/'missing.zip')
            result=subprocess.run(['bash', str(root/'scripts/train_matrix.sh')],
                                  env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Warm-start checkpoint', result.stderr)
            self.assertFalse((root/'trace.jsonl').exists())

    def test_busy_data_port_selects_next_complete_set_in_both_commands_and_name(self):
        start = find_free_offset(3000, 3100)
        with socket.socket() as occupied:
            # Blocking a data port also blocks several successive offsets.
            occupied.bind(('127.0.0.1', BASE_PORTS[-1] + start))
            occupied.listen()
            selected = find_free_offset(start, 3200)
            self.assertGreaterEqual(selected, start + 8)
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder); env=self.prepare(root); env['PORT_OFFSET']=str(start)
                result=subprocess.run(['bash', str(root/'scripts/train_matrix.sh'), '--dry-run'],
                                      env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                commands=[shlex.split(line.split(': ', 1)[1]) for line in result.stdout.splitlines()
                          if line.startswith(('Simulator command:', 'Training command:'))]
                self.assertEqual(len(commands), 2)
                for command in commands:
                    self.assertEqual(command[command.index('--port-offset')+1], str(selected))
                self.assertIn(f'_offset{selected}', result.stdout)
                self.assertFalse((root/'trace.jsonl').exists())

    def test_no_free_port_set_fails_clearly_and_closes_probe_sockets(self):
        start=find_free_offset(3300, 3400)
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', BASE_PORTS[-1] + start))
            occupied.listen()
            with self.assertRaisesRegex(RuntimeError, 'No complete simulator port set'):
                find_free_offset(start, start + 7)
            # Failed probes must release the earlier, free ports.
            with socket.socket() as probe:
                probe.bind(('0.0.0.0', BASE_PORTS[0] + start))
        self.assertEqual(find_free_offset(start, start), start)

    def test_camera_topic_404_is_retried_before_sdk_subscription(self):
        topic='http://127.0.0.1:7521/robot/map_0/vehicle_0/sensor/camera/front_center/jpeg/'
        events=[]
        robot=MagicMock()
        robot.camera.capture.return_value=np.zeros((32,32,3), np.uint8)
        def request(req, **kwargs):
            self.assertEqual(req.full_url, topic)
            self.assertEqual(req.get_method(), 'HEAD')
            events.append('probe')
            if events.count('probe') <= 2:
                raise HTTPError(topic, 404, 'Not Found', {}, None)
            return MagicMock()
        def factory(*args, **kwargs):
            events.append('subscribe')
            return robot
        with patch('scripts.wait_matrix.urlopen', side_effect=request), \
             patch('scripts.wait_matrix.time.sleep'), \
             patch('scripts.wait_matrix.DB21J', side_effect=factory):
            wait_for_camera('map_0/vehicle_0', 7521, 5.)
        self.assertEqual(events, ['probe', 'probe', 'probe', 'subscribe'])
        robot.camera.start.assert_called_once()

    def test_missing_camera_topic_times_out_without_starting_sdk(self):
        clock=[0.]
        def sleep(seconds): clock[0] += seconds
        with patch('scripts.wait_matrix.urlopen', side_effect=HTTPError('http://example', 404, 'Not Found', {}, None)), \
             patch('scripts.wait_matrix.time.monotonic', side_effect=lambda: clock[0]), \
             patch('scripts.wait_matrix.time.sleep', side_effect=sleep), \
             patch('scripts.wait_matrix.DB21J') as factory:
            with self.assertRaisesRegex(RuntimeError, 'camera topic did not become ready'):
                wait_for_camera('map_0/vehicle_0', 7521, .5)
        factory.assert_not_called()

    def test_ready_check_only_uses_camera_at_requested_port(self):
        robot=MagicMock()
        robot.camera.capture.side_effect=[None, np.zeros((32,32,3), np.uint8)]
        with patch('scripts.wait_matrix.urlopen'), patch('scripts.wait_matrix.DB21J', return_value=robot) as factory:
            wait_for_camera('map_0/vehicle_0', 7521, 5.)
        factory.assert_called_once_with('map_0/vehicle_0', simulated=True, port=7521)
        robot.motors.start.assert_not_called()
        robot.motors.set_pwm.assert_not_called()
        robot.camera.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
