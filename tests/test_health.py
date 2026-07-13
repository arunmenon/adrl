"""Runtime rung-health circuit breaker."""

from router.health import RungHealthMonitor


def test_failure_opens_circuit_and_success_closes_it():
    now = [10.0]
    monitor = RungHealthMonitor(clock=lambda: now[0], cooldown_s=30)
    monitor.record_failure("local-code", "timeout")
    assert monitor.snapshot()["local"]["healthy"] is False
    assert monitor.state("local").last_error == "timeout"
    monitor.record_success("local")
    assert monitor.snapshot()["local"]["healthy"] is True


def test_cooldown_moves_unhealthy_rung_to_half_open():
    now = [10.0]
    monitor = RungHealthMonitor(clock=lambda: now[0], cooldown_s=30)
    monitor.record_failure("local")
    now[0] = 39.9
    assert monitor.snapshot()["local"]["healthy"] is False
    now[0] = 40.0
    assert monitor.snapshot()["local"]["healthy"] is True


def test_external_probe_hook_updates_registry():
    monitor = RungHealthMonitor()
    monitor.set_health("cheap-cloud", False, "rate limited")
    assert monitor.snapshot()["cheap_cloud"]["healthy"] is False
    monitor.set_health("cheap-cloud", True)
    assert monitor.snapshot()["cheap_cloud"]["healthy"] is True
