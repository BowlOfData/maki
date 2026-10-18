#!/usr/bin/env python3
"""
Verified workflow example: maker/checker with bounded revision.

A task with ``verify=VerifySpec(...)`` only counts as COMPLETED once an independent
verifier accepts its result. A REVISE verdict re-runs the task with the verifier's
feedback (``__verification_feedback`` / ``__previous_result`` in its context), up to
``max_revisions`` times; otherwise the task fails and its dependents are skipped.

Runs offline with scripted stub agents (no LLM needed).
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maki.agents import AgentManager, Verdict, VerifyResult, VerifySpec, WorkflowTask  # noqa: E402


class ScriptedAgent:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def execute_task_with_retry(self, task, context=None, max_retries=3, retry_delay=1.0):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        if context and context.get("__verification_feedback"):
            print(f"  maker got feedback: {context['__verification_feedback']}")
        return out


def must_contain_total(ctx):
    """Deterministic checker: cheap checks first, cite evidence, tell the maker how to fix it."""
    if "total:" in ctx["result"]:
        return VerifyResult(Verdict.ACCEPT, evidence="found a 'total:' line")
    return VerifyResult(Verdict.REVISE, evidence="no 'total:' line",
                        fixes="end the report with a line 'total: <number>'")


def main():
    mgr = AgentManager(MagicMock())
    mgr.agents["writer"] = ScriptedAgent(["Sales were good.", "Sales were good.\ntotal: 42"])
    mgr.agents["publisher"] = ScriptedAgent(["published"])

    workflow = [
        WorkflowTask("report", "writer", "Write the sales report",
                     verify=VerifySpec(must_contain_total, max_revisions=2)),
        WorkflowTask("publish", "publisher", "Publish the report", dependencies=["report"]),
    ]
    mgr.run_workflow(workflow)

    for name, entry in mgr.last_workflow_state.tasks.items():
        v = entry.get("verification", {})
        print(f"{name}: {entry['status'].value} verdict={v.get('verdict')} revisions={v.get('revisions')}")

    # For an LLM-based checker use a *different* agent:
    #   from maki.agents import agent_verifier
    #   VerifySpec(agent_verifier("reviewer", "The report must cite its sources."))


if __name__ == "__main__":
    main()
