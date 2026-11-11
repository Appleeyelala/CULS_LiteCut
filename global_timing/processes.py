"""Terminate owned Linux process trees, including separately sessioned tools."""
import os
from pathlib import Path
import signal


def terminate_tree(pid: int):
    """Freeze discovery before killing, so a timed-out case leaves no tool running.

    Only descendants of the supplied owned subprocess are signalled. Slurm also
    provides a job cgroup boundary; this helper handles individual case timeouts.
    """
    owned = {pid}

    def send(target, sig):
        try:
            os.kill(target, sig)
        except ProcessLookupError:
            pass

    send(pid, signal.SIGSTOP)
    while True:
        added = set()
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                # comm may contain spaces or parentheses; fields after final ')'.
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                child, parent = int(entry.name), int(fields[1])
            except (OSError, ValueError, IndexError):
                continue
            if parent in owned and child not in owned:
                send(child, signal.SIGSTOP)
                added.add(child)
        if not added:
            break
        owned.update(added)
    for child in owned - {pid}:
        send(child, signal.SIGKILL)
    send(pid, signal.SIGKILL)
    return sorted(owned)
