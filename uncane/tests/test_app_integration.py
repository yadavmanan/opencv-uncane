import copy

import numpy as np

from uncane.edge import app as edge_app
from uncane.edge.config import DEFAULTS


class StubDepth:
    def __init__(self, *_args):
        pass

    def infer(self, _frame):
        return np.ones((16, 16), dtype=np.float32)


def test_app_routes_local_commands_and_persists_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(edge_app, "DepthEngine", StubDepth)
    cfg = copy.deepcopy(DEFAULTS)
    cfg["detector"]["enabled"] = False
    cfg["flow"]["enabled"] = False
    cfg["cloud"]["enabled"] = False
    cfg["audio"]["enabled"] = False
    cfg["voice"]["backend"] = "none"
    cfg["debug"]["show_windows"] = False
    cfg["monitor"]["enabled"] = False
    cfg["blackbox"]["enabled"] = False
    cfg["user"]["profile_path"] = str(tmp_path / "profile.json")
    instance = edge_app.EdgeApp(cfg, headless=True)
    forwarded = []
    instance.ask = forwarded.append

    try:
        instance.handle_utterance("indoor mode")
        instance.handle_utterance("quiet mode on")
        instance.handle_utterance("add note meet Alex at 4")
        instance.handle_utterance("what is ahead")

        assert instance.cfg["hazards"]["critical_m"] == cfg["environment"]["indoor_critical_m"]
        assert instance.arbiter.quiet_mode is True
        assert instance.profile.data["notes"] == ["meet alex at 4"]
        assert forwarded == ["what is ahead"]
    finally:
        instance.speaker.close()
        instance.beeper.close()
        instance._flow_pool.shutdown(wait=True)
