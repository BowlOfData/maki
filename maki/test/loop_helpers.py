"""Shared fakes for the agent-loop tests."""
import json


class FakeAgent:
    """Maker/judge stand-in returning scripted replies (the last one repeats)."""

    def __init__(self, replies, stateful=True):
        self.replies, self.stateful = list(replies), stateful
        self.prompts, self.resets = [], 0
        self.plugins, self.allow_dangerous_tools = {}, False

    def execute_task_with_retry(self, task, context=None, use_plugins=False, **kw):
        self.prompts.append(task)
        out = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        if isinstance(out, Exception):
            raise out
        return out

    def reset_conversation(self):
        self.resets += 1


def verdict(passing, block="", **feedback):
    """Judge reply: ``passing`` maps requirement id -> bool."""
    return json.dumps({
        "results": [{"id": k, "pass": v, "feedback": feedback.get(k, "" if v else "fix it")}
                    for k, v in passing.items()],
        "block": block,
    })


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
