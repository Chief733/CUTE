"""Narrow, version-pinned adapter for MindSpore 2.4's framework dumper.

AscendProfiler.stop() flushes the dumper but leaves its FILE* map alive. Every
new output directory otherwise retains another fd and potentially unlinked
tmpfs storage. UnInit closes that map after collection is stopped; the next
AscendProfiler.start() calls Init again. These are exported C++ ABI symbols,
not a supported MindSpore Python API. Revalidate before changing the wheel.
"""
import ctypes
import hashlib
from pathlib import Path

import mindspore as ms


class DumperCleanup:
    def __init__(self):
        if ms.__version__ != '2.4.0':
            raise RuntimeError('Native dumper cleanup requires verified MindSpore 2.4.0')
        library = Path(ms.__file__).parent/'lib/libmindspore_common.so'
        expected = '1cb5c68f04841248dccbdfd74d9429aefa5b61ee9464e7212da435ad688bf76a'
        if hashlib.sha256(library.read_bytes()).hexdigest() != expected:
            raise RuntimeError('MindSpore native library changed; revalidate the dumper ABI before enabling this adapter')
        self.library = ctypes.CDLL(str(library))
        prefix = '_ZN9mindspore8profiler6ascend19ProfilingDataDumper'
        self.instance = getattr(self.library, prefix+'11GetInstanceEv')
        self.instance.argtypes = []
        self.instance.restype = ctypes.c_void_p
        self.uninit = getattr(self.library, prefix+'6UnInitEv')
        self.uninit.argtypes = [ctypes.c_void_p]
        self.uninit.restype = None

    def close_after_stop(self):
        self.uninit(self.instance())
