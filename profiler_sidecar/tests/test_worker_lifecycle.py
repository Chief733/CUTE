"""Black-box daemon restart, singleton and disk-pressure tests (CPU only)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import atomic_json, process_identity, read_json, require_worker

WORKER=Path(__file__).resolve().parents[1]/'worker.py'


def wait_for(predicate, timeout=12):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError('Timed out waiting for daemon state')


class WorkerLifecycleTests(unittest.TestCase):
    def prepare(self, directory, min_disk=0.001):
        root=Path(directory)/'ram'
        archive=Path(directory)/'archive'
        root.mkdir();archive.mkdir()
        config={'archive':str(archive),'launcher':process_identity(),'parse':False,
                'min_disk_free_gib':min_disk}
        atomic_json(root/'config.json',config)
        segment=root/'raw/segment_000000/rank_0'
        segment.mkdir(parents=True)
        (segment/'data').write_bytes(b'real raw bytes'*100)
        atomic_json(segment/'ready.json',{'rank':0,'segment':0})
        return root,archive,segment

    def start(self,root):
        proc=subprocess.Popen([sys.executable,str(WORKER),'run',str(root)],
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: self.stop(proc))
        return proc

    @staticmethod
    def stop(proc):
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)

    def test_singleton_restart_and_detached_drain(self):
        with tempfile.TemporaryDirectory() as td:
            root,archive,raw=self.prepare(td)
            first=self.start(root)
            wait_for(lambda: (archive/'segments/segment_000000/rank_0/archive.json').exists())
            wait_for(lambda: not raw.exists())
            second=self.start(root)
            self.assertNotEqual(second.wait(timeout=5),0)
            self.assertIsNone(first.poll())
            first.terminate();first.wait(timeout=5)
            with self.assertRaises(RuntimeError):
                require_worker(root)
            resumed=self.start(root)
            wait_for(lambda: (root/'status.json').exists() and
                     read_json(root/'status.json')['process']['pid']==resumed.pid)
            atomic_json(root/'launcher_done.json',{'returncode':0})
            self.assertEqual(resumed.wait(timeout=8),0)
            result=read_json(archive/'status.json')
            self.assertEqual(result['state'],'complete')
            self.assertEqual(result['archived_segments'],1)

    def test_disk_pressure_preserves_raw_and_surfaces_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root,archive,raw=self.prepare(td,min_disk=1e9)
            worker=self.start(root)
            wait_for(lambda: (root/'status.json').exists() and read_json(root/'status.json')['fatal'])
            with self.assertRaises(RuntimeError):
                require_worker(root)
            self.assertTrue((raw/'data').exists())
            atomic_json(root/'launcher_done.json',{'returncode':1})
            self.assertEqual(worker.wait(timeout=8),0)
            self.assertEqual(read_json(archive/'status.json')['state'],'incomplete')
            self.assertTrue((raw/'data').exists())


if __name__=='__main__':
    unittest.main()
