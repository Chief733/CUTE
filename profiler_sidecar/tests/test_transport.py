import hashlib
import json
from collections import namedtuple
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GIB, atomic_json, capture_capacity_problem, trace_events
from worker import archive_segment
from launch import cache_specification


class TransportTests(unittest.TestCase):
    def test_capture_does_not_require_a_live_or_successful_worker(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            config={'max_ram_gib':64,'min_ram_free_gib':8,'tmpfs_used_at_start':0}
            usage=namedtuple('Usage','total used free')(256*GIB,GIB,255*GIB)
            with patch('common.shutil.disk_usage',return_value=usage):
                self.assertIsNone(capture_capacity_problem(root,config))
                atomic_json(root/'status.json',{'process':None,'heartbeat':0,
                                               'fatal':'disk stalled','raw_bytes':0})
                self.assertIsNone(capture_capacity_problem(root,config))
                (root/'status.json').write_text('invalid metadata')
                self.assertIsNone(capture_capacity_problem(root,config))

    def test_buffer_limit_still_detected_with_a_frozen_worker(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            config={'max_ram_gib':64,'min_ram_free_gib':8,'tmpfs_used_at_start':0}
            atomic_json(root/'status.json',{'raw_bytes':GIB,'tmpfs_used_bytes':2*GIB})
            usage=namedtuple('Usage','total used free')
            with patch('common.shutil.disk_usage',return_value=usage(256*GIB,66*GIB,190*GIB)):
                self.assertIn('backlog limit',capture_capacity_problem(root,config))
            with patch('common.shutil.disk_usage',return_value=usage(256*GIB,250*GIB,6*GIB)):
                self.assertIn('reserve reached',capture_capacity_problem(root,config))

    def test_cache_identity_ignores_output_and_profiling_but_tracks_shapes(self):
        first={'context':{'enable_compile_cache':True},'profile':True,'output_dir':'first',
               'model':{'seq_length':4096}}
        other=dict(first,profile=False,output_dir='second')
        self.assertEqual(cache_specification(first,8),cache_specification(other,8))
        self.assertEqual(
            cache_specification(first,8)['sidecar_graph_instrumentation'],
            'profiler_preinit_step_trace_v1',
        )
        other['model']={'seq_length':2048}
        self.assertNotEqual(cache_specification(first,8),cache_specification(other,8))

    def test_custom_training_entries_have_isolated_cache_identities(self):
        config={'context':{'enable_compile_cache':True},'model':{'seq_length':4096}}
        with tempfile.TemporaryDirectory() as td:
            first=Path(td)/'first.py'
            second=Path(td)/'second.py'
            first.write_text('print("first")\n')
            second.write_text('print("second")\n')
            first_spec=cache_specification(
                config,8,first,{'CYF_COMPARE_MODE':'offload'})
            second_spec=cache_specification(
                config,8,second,{'CYF_COMPARE_MODE':'offload'})
            control_spec=cache_specification(
                config,8,first,{'CYF_COMPARE_MODE':'control'})
            self.assertNotEqual(first_spec,second_spec)
            self.assertNotEqual(first_spec,control_spec)
            self.assertNotIn('sidecar_training_entry',cache_specification(config,8))
    def test_commit_before_retiring_ram_and_retry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root/'raw'/'segment_000001'/'rank_0'
            raw.mkdir(parents=True)
            data = b'raw-profiler-data\x00' * 100000
            (raw/'data').write_bytes(data)
            atomic_json(raw/'ready.json', {'rank':0,'segment':1})
            with patch('worker.os.fsync', side_effect=OSError('injected disk error')):
                with self.assertRaises(OSError):
                    archive_segment(raw,root/'archive')
            self.assertTrue((raw/'data').exists())
            self.assertFalse((root/'archive/segments/segment_000001/rank_0').exists())
            record = archive_segment(raw,root/'archive')
            target = root/'archive/segments/segment_000001/rank_0'
            self.assertEqual((target/'data').read_bytes(),data)
            item = next(x for x in record['files'] if x['path']=='data')
            self.assertEqual(item['sha256'],hashlib.sha256(data).hexdigest())
            self.assertFalse(raw.exists())
            # Crash after commit / before RAM unlink must be recoverable.
            raw.mkdir(parents=True)
            atomic_json(raw/'ready.json', {'rank':0,'segment':1})
            archive_segment(raw,root/'archive')
            self.assertFalse(raw.exists())

    def test_fast_mode_retains_ram_after_archive_commit(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            raw=root/'raw/segment_000001/rank_0'
            raw.mkdir(parents=True)
            (raw/'data').write_bytes(b'immutable')
            atomic_json(raw/'ready.json',{'rank':0,'segment':1})
            archive_segment(raw,root/'archive',retire_source=False)
            self.assertTrue((raw/'data').is_file())
            # Restarting the archive phase must honor the same second latch.
            archive_segment(raw,root/'archive',retire_source=False)
            self.assertTrue((raw/'data').is_file())

    def test_trace_streaming_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'trace.json'
            events = [{'name':'中文'*40000,'ph':'X','dur':2}, {'name':'kernel','ph':'X'}]
            path.write_text(json.dumps(events,ensure_ascii=False))
            self.assertEqual(list(trace_events(path)),events)
            for data in ('[{"name":"truncated"}', '[{},]', '[{}]x', '{}'):
                path.write_text(data)
                with self.assertRaises(ValueError):
                    list(trace_events(path))


if __name__ == '__main__':
    unittest.main()
