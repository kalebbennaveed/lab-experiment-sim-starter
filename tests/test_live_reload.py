import json
from pathlib import Path
import shutil
import time

from lab_sim.server import ExperimentRunner, file_revision

ROOT = Path(__file__).resolve().parents[1]


def await_state(runner, state, different_from=None):
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        status = runner.status()
        if status["state"] == state and (different_from is None or status["result_revision"] != different_from):
            return status
        time.sleep(.05)
    raise AssertionError(f"Runner did not reach {state}: {runner.status()}")


def test_source_edit_error_recovery_rerun_and_mode_change(tmp_path):
    for name in ("src", "examples", "config", "web"):
        shutil.copytree(ROOT/name,tmp_path/name,ignore=shutil.ignore_patterns("__pycache__"))
    runner = ExperimentRunner(tmp_path,tmp_path/"examples/lab_demo.py",tmp_path/"config/lab.toml",duration=.2,mode="lab")
    runner.start()
    try:
        initial = await_state(runner,"ready")
        before = json.loads(runner.result)
        # Edit an imported module, not only the entrypoint.
        module = tmp_path/"src/lab_sim/trajectory.py"
        source = module.read_text()
        module.write_text(source.replace("self.offset = np.asarray(config.offset, dtype=float)","self.offset = np.asarray(config.offset, dtype=float) + np.array([0.25, 0, 0])"))
        edited = await_state(runner,"ready",different_from=initial["result_revision"])
        after = json.loads(runner.result)
        assert after["history"]["reference"][0][0] == before["history"]["reference"][0][0]+.25
        successful_result = runner.result
        module.write_text("this is broken Python !!!\n")
        failed = await_state(runner,"error")
        assert "SyntaxError" in failed["error"]
        assert runner.result == successful_result
        module.write_text(source)
        recovered = await_state(runner,"ready",different_from=edited["result_revision"])
        runner.request_run()
        rerun = await_state(runner,"ready",different_from=recovered["result_revision"])
        runner.select_experiment("generic","crown")
        await_state(runner,"ready",different_from=rerun["result_revision"])
        assert not json.loads(runner.result)["config"]["lab"]["enabled"]
        assert runner.status()["pattern"] == "crown"
    finally:
        runner.close()


def test_web_revision_changes_when_asset_changes(tmp_path):
    asset = tmp_path/"app.js"
    asset.write_text("first")
    before = file_revision([asset])
    asset.write_text("second")
    assert file_revision([asset]) != before
