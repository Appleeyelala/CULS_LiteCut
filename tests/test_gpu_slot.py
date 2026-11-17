import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from global_timing.gpu_slot import acquire_gpu_slot


def holder(root, acquired, release):
    with patch.dict(os.environ, {'SLURM_JOB_ID': 'test', 'CUDA_VISIBLE_DEVICES': '0'}):
        handle, _ = acquire_gpu_slot(Path(root))
        acquired.set()
        release.wait(10)
        handle.close()


class GpuSlotTests(unittest.TestCase):
    def test_shared_slot_waits_and_releases(self):
        ctx = mp.get_context('spawn')
        with tempfile.TemporaryDirectory() as root:
            a, b, release_a, release_b = [ctx.Event() for _ in range(4)]
            first = ctx.Process(target=holder, args=(root, a, release_a))
            second = ctx.Process(target=holder, args=(root, b, release_b))
            first.start()
            try:
                self.assertTrue(a.wait(5))
                second.start()
                self.assertFalse(b.wait(0.25))
                release_a.set()
                self.assertTrue(b.wait(5))
            finally:
                release_a.set(); release_b.set()
                first.join(10)
                if second.pid is not None:
                    second.join(10)
            self.assertEqual(first.exitcode, 0)
            self.assertEqual(second.exitcode, 0)
