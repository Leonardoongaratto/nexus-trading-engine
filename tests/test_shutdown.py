import subprocess
import sys

import pytest


def test_run_server_terminates_on_sigint() -> None:
    code = """
import time
import signal
import threading
from xau_bot.config import Settings
from xau_bot.server import run_server

def trigger_sig():
    time.sleep(0.5)
    signal.raise_signal(signal.SIGINT)

threading.Thread(target=trigger_sig, daemon=True).start()
run_server(Settings(server_port=0))
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail("Server deadlocked on SIGINT, did not terminate within 3 seconds")

    assert proc.returncode == 0
