"""Tests for _ControllerBase.close() and its weakref.finalize-based process kill.

Exercises only the close()/finalizer wiring in controller.py. Runtime/COM
construction (_Controller400/500/503.__init__) is unchanged pre-existing
code and is not re-tested here.
"""

import weakref

import pytest

from rivia.controller import controller


@pytest.fixture
def fake_kill_process(monkeypatch):
    """Replace controller.kill_process with a call-recording stub."""
    calls = []

    def _fake(pid):
        calls.append(pid)
        return pid is not None

    monkeypatch.setattr(controller, "kill_process", _fake)
    return calls


def _make_bare_controller(pid):
    """Build a _ControllerBase instance with a finalizer wired like the
    real subclass constructors do, without running __init__ (which
    requires a live COM object)."""
    obj = controller._ControllerBase.__new__(controller._ControllerBase)
    obj._finalizer = weakref.finalize(obj, controller.kill_process, pid)
    return obj


class TestClose:
    def test_close_invokes_kill_process_with_pid(self, fake_kill_process):
        obj = _make_bare_controller(1234)
        obj.close()
        assert fake_kill_process == [1234]

    def test_close_is_idempotent(self, fake_kill_process):
        obj = _make_bare_controller(1234)
        obj.close()
        obj.close()
        assert fake_kill_process == [1234]

    def test_close_with_none_pid_calls_through_safely(self, fake_kill_process):
        obj = _make_bare_controller(None)
        obj.close()
        assert fake_kill_process == [None]

    def test_close_without_finalizer_is_noop(self, fake_kill_process):
        obj = controller._ControllerBase.__new__(controller._ControllerBase)
        obj.close()  # no _finalizer attribute set at all
        assert fake_kill_process == []

    def test_gc_triggers_same_finalizer(self, fake_kill_process):
        obj = _make_bare_controller(5678)
        del obj
        assert fake_kill_process == [5678]

    def test_explicit_close_then_gc_kills_only_once(self, fake_kill_process):
        obj = _make_bare_controller(9999)
        obj.close()
        del obj
        assert fake_kill_process == [9999]
