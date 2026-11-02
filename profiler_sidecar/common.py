"""Small, MindSpore-free helpers shared by the producer and sidecar."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

GIB = 1024 ** 3


def atomic_json(path, value, durable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
            if durable:
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(tmp, path)
        if durable:
            sync_directory(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path):
    return json.loads(Path(path).read_text())


def _identity_from_proc(proc_pid, expected_pid=None):
    """Read identity from a host-visible /proc entry.

    Some container launchers expose a host-mounted /proc while getpid()/Popen
    return PID-namespace IDs.  Persist both IDs so liveness checks remain exact.
    """
    try:
        # The process name may contain spaces or parentheses.
        proc_pid = int(proc_pid)
        stat_text = Path(f'/proc/{proc_pid}/stat').read_text()
        stat = stat_text.rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        status = Path(f'/proc/{proc_pid}/status').read_text()
        nspid = next((line.split()[1:] for line in status.splitlines()
                      if line.startswith('NSpid:')), ())
        pid = int(nspid[-1]) if nspid else int(stat_text.split(' ', 1)[0])
        if expected_pid is not None and pid != int(expected_pid):
            return None
        return {'pid': pid, 'proc_pid': proc_pid, 'start_ticks': int(stat[19])}
    except (OSError, ValueError, IndexError):
        return None


def _proc_pid_for_namespace_pid(pid):
    """Resolve a namespace PID only on the uncommon host-/proc mismatch path."""
    pid = int(pid)
    try:
        namespace = os.stat('/proc/self/ns/pid').st_ino
    except OSError:
        return None

    direct = Path(f'/proc/{pid}')
    try:
        if os.stat(direct/'ns/pid').st_ino == namespace:
            identity = _identity_from_proc(pid, pid)
            if identity:
                return pid
    except OSError:
        pass

    # Popen children are the hot path.  /proc task children contains the
    # host-visible IDs even when Popen.pid is namespaced.
    try:
        for children in Path('/proc/self/task').glob('*/children'):
            for candidate in children.read_text().split():
                try:
                    if os.stat(f'/proc/{candidate}/ns/pid').st_ino != namespace:
                        continue
                except OSError:
                    continue
                if _identity_from_proc(candidate, pid):
                    return int(candidate)
    except OSError:
        pass

    # Persisted identities created by older code lack proc_pid.  This fallback
    # is deliberately last; new records never need a full /proc scan.
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if os.stat(entry/'ns/pid').st_ino != namespace:
                continue
        except OSError:
            continue
        if _identity_from_proc(entry.name, pid):
            return int(entry.name)
    return None


def process_identity(pid=None):
    if pid is None or int(pid) == os.getpid():
        try:
            proc_pid = int(Path('/proc/self/stat').read_text().split(' ', 1)[0])
        except (OSError, ValueError):
            return None
        return _identity_from_proc(proc_pid, os.getpid())
    pid = int(pid)
    proc_pid = _proc_pid_for_namespace_pid(pid)
    return _identity_from_proc(proc_pid, pid) if proc_pid is not None else None


def is_alive(identity):
    if not identity:
        return False
    try:
        if identity.get('proc_pid') is not None:
            current = _identity_from_proc(identity['proc_pid'], identity['pid'])
        else:
            current = process_identity(identity['pid'])
        return bool(current) and current['pid'] == identity['pid'] and (
            current['start_ticks'] == identity['start_ticks'])
    except (KeyError, TypeError, ValueError):
        return False


def tree_bytes(path):
    total = 0
    # os.walk tolerates a directory being removed before its scandir call;
    # pathlib.rglob in Python 3.10 raises outside the per-file try/except.
    for directory, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(directory)/name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def raw_rank_paths(root):
    result=[]
    try:
        segments=list(Path(root).iterdir())
    except FileNotFoundError:
        return result
    for segment in segments:
        if not segment.name.startswith('segment_'):
            continue
        try:
            result.extend(p for p in segment.iterdir() if p.name.startswith('rank_') and p.is_dir())
        except FileNotFoundError:
            continue
    return result


def require_worker(root, max_age=30):
    status = read_json(Path(root) / 'status.json')
    if not is_alive(status.get('process')) or time.time() - status['heartbeat'] > max_age:
        raise RuntimeError('Profiler sidecar is not alive/responsive; raw segments are preserved in ' + str(root))
    if status.get('fatal'):
        raise RuntimeError('Profiler sidecar failed: ' + status['fatal'])
    return status


def capture_capacity_problem(root, config):
    """Read RAM state without depending on a worker heartbeat or acknowledgement.

    When the worker is stalled, tmpfs growth since its last measurement estimates
    newly queued data. The per-run budget is a soft limit; filesystem free space
    is checked independently even if worker status is absent or unreadable.
    """
    usage = shutil.disk_usage(root)
    if usage.free < config['min_ram_free_gib'] * GIB:
        return 'Profiler tmpfs reserve reached; new capture disabled, training continues'
    try:
        status = read_json(Path(root) / 'status.json')
    except (OSError, ValueError):
        status = {}
    measured_used = status.get('tmpfs_used_bytes', config.get('tmpfs_used_at_start', usage.used))
    estimate = status.get('raw_bytes', 0) + max(0, usage.used - measured_used)
    if estimate >= config['max_ram_gib'] * GIB:
        return 'Profiler RAM backlog limit reached; new capture disabled, training continues'
    return None


def trace_events(path):
    """Validate/stream a Chrome trace array without loading gigabytes into RAM."""
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        buffer, pos, eof = '', 0, False

        def fill():
            nonlocal buffer, pos, eof
            buffer = buffer[pos:] + stream.read(65536)
            pos = 0
            eof = len(buffer) == 0 or stream.tell() == os.fstat(stream.fileno()).st_size

        def token():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer):
                    return buffer[pos]
                if eof:
                    raise ValueError('Unexpected end of trace')
                fill()

        fill()
        if token() != '[':
            raise ValueError('Expected a Chrome trace array')
        pos += 1
        first = True
        while token() != ']':
            if not first:
                if token() != ',':
                    raise ValueError('Expected comma in trace')
                pos += 1
            token()
            while True:
                try:
                    event, pos = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    if eof or len(buffer) - pos > 64 * 1024 * 1024:
                        raise
                    fill()
            if not isinstance(event, dict):
                raise ValueError('Trace event is not an object')
            yield event
            first = False
        pos += 1
        if (buffer[pos:] + stream.read()).strip():
            raise ValueError('Trailing data after trace array')
