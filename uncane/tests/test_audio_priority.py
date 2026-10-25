import threading
import time

from uncane.edge.audio_feedback import PRIORITY_CLOUD, PRIORITY_INFO, PRIORITY_SAFETY, Speaker


class FakeSpeaker(Speaker):
    """Each utterance 'plays' until it is interrupted or `release` is set."""

    def __init__(self):
        self.events = []
        self.release = threading.Event()
        super().__init__()

    def _make_speak(self):
        def speak(text, aborted):
            self.events.append(("start", text))
            while not aborted() and not self.release.is_set():
                time.sleep(0.002)
            self.events.append(("cut" if aborted() else "done", text))

        return speak


def wait_for(speaker, event, timeout=2.0):
    end = time.monotonic() + timeout
    while event not in speaker.events:
        assert time.monotonic() < end, speaker.events
        time.sleep(0.002)


def test_cloud_answer_replaces_status_chatter():
    s = FakeSpeaker()
    s.say("Looking.", PRIORITY_INFO)
    wait_for(s, ("start", "Looking."))
    s.say("still looking", PRIORITY_INFO)
    s.say("The bottle is on your left.", PRIORITY_CLOUD)
    wait_for(s, ("start", "The bottle is on your left."))
    s.release.set()
    wait_for(s, ("done", "The bottle is on your left."))
    assert ("start", "still looking") not in s.events
    s.close()


def test_status_never_cuts_off_cloud_answer():
    s = FakeSpeaker()
    s.say("The bottle is on your left.", PRIORITY_CLOUD)
    wait_for(s, ("start", "The bottle is on your left."))
    s.say("Camera image is dark.", PRIORITY_INFO)
    time.sleep(0.05)
    assert ("cut", "The bottle is on your left.") not in s.events
    s.release.set()
    wait_for(s, ("done", "Camera image is dark."))
    s.close()


def test_safety_interrupts_and_cloud_answer_is_replayed():
    s = FakeSpeaker()
    s.say("The bottle is on your left.", PRIORITY_CLOUD)
    wait_for(s, ("start", "The bottle is on your left."))
    s.say("Stop. Chair ahead.", PRIORITY_SAFETY)
    wait_for(s, ("start", "Stop. Chair ahead."))
    assert ("cut", "The bottle is on your left.") in s.events
    s.release.set()
    wait_for(s, ("done", "The bottle is on your left."))
    assert s.events.index(("done", "Stop. Chair ahead.")) < s.events.index(("done", "The bottle is on your left."))
    s.close()


def test_stop_clears_queue_and_does_not_replay():
    s = FakeSpeaker()
    s.say("The bottle is on your left.", PRIORITY_CLOUD)
    wait_for(s, ("start", "The bottle is on your left."))
    s.say("Device ready.", PRIORITY_INFO)
    s.stop()
    wait_for(s, ("cut", "The bottle is on your left."))
    s.release.set()
    time.sleep(0.1)
    assert ("start", "Device ready.") not in s.events
    assert s.events.count(("start", "The bottle is on your left.")) == 1
    s.close()
