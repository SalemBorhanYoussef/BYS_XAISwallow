import os
import threading
import subprocess
import signal
import time
from typing import Optional, List, Dict, Any


class TrainingManager:
    """Manage a single background training process and capture logs.

    - start(cmd, cwd): launches a subprocess with unbuffered output
    - stop(): terminates the process
    - status(): returns dict with running, pid, returncode, uptime
    - get_logs(since): returns new log lines since a cursor
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._log_lines: List[str] = []
        self._log_lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._cwd: Optional[str] = None
        self._cmd: Optional[List[str]] = None

    def start(self, cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if self._proc and self._proc.poll() is None:
            return {"ok": False, "error": "Training already running", "pid": self._proc.pid}

        # reset state
        with self._log_lock:
            self._log_lines.clear()
        self._start_time = time.time()
        self._cwd = cwd or os.getcwd()
        self._cmd = cmd

        # Ensure unbuffered Python output
        env_final = os.environ.copy()
        if env:
            env_final.update(env)
        env_final.setdefault("PYTHONUNBUFFERED", "1")

        creationflags = 0
        if os.name == "nt":
            # Create new process group on Windows to allow clean termination
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        self._proc = subprocess.Popen(
            cmd,
            cwd=self._cwd,
            env=env_final,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            creationflags=creationflags,
        )

        def _reader():
            assert self._proc is not None
            if self._proc.stdout is None:
                return
            for line in self._proc.stdout:
                with self._log_lock:
                    self._log_lines.append(line.rstrip("\n"))
            # Drain any remaining output
            rest = self._proc.stdout.read()
            if rest:
                for line in rest.splitlines():
                    with self._log_lock:
                        self._log_lines.append(line)

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

        return {"ok": True, "pid": self._proc.pid, "cwd": self._cwd, "cmd": cmd}

    def stop(self) -> Dict[str, Any]:
        if not self._proc or self._proc.poll() is not None:
            return {"ok": True, "stopped": False, "reason": "no process"}
        try:
            if os.name == "nt":
                # Send CTRL-BREAK to the process group (if possible)
                try:
                    self._proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                    time.sleep(0.5)
                except Exception:
                    pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        finally:
            rc = self._proc.poll()
            return {"ok": True, "stopped": True, "returncode": rc}

    def status(self) -> Dict[str, Any]:
        running = self._proc is not None and self._proc.poll() is None
        return {
            "running": running,
            "pid": (self._proc.pid if self._proc else None),
            "returncode": (self._proc.poll() if self._proc else None),
            "uptime_sec": (time.time() - self._start_time) if (self._start_time and running) else 0.0,
            "cwd": self._cwd,
            "cmd": self._cmd,
            "log_size": len(self._log_lines),
        }

    def get_logs(self, since: int = 0, max_lines: int = 500) -> Dict[str, Any]:
        with self._log_lock:
            total = len(self._log_lines)
            start = max(0, min(since, total))
            end = min(total, start + max(1, max_lines))
            lines = self._log_lines[start:end]
        return {"from": start, "to": end, "total": total, "lines": lines}


# Global singleton for app
training_manager = TrainingManager()
