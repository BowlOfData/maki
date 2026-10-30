"""
Run a loop on a remote Maki server (see examples/loop_server.yaml) and watch it.

    maki serve --config examples/loop_server.yaml --port 8123
    python examples/loop_remote.py [http://127.0.0.1:8123]

Requirements are plain text: a server never runs commands a caller sends. The server's own
judge agent decides when they are met. Returning ``Decision.STOP`` from the hook cancels the
remote loop after its current iteration.
"""
import sys

from maki.agents import Decision
from maki.distributed.proxy import AgentProxy


def main() -> None:
    proxy = AgentProxy(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123", timeout=120)

    def show(event):
        failing = [k for k, v in event["results"].items() if not v["passed"]]
        print(f"iteration {event['iteration']}: {event['passing']}/{event['total']} passing, failing={failing}")
        return Decision.CONTINUE

    result = proxy.run_loop(
        "Write a short product description for the 'Aurora' reusable water bottle "
        "(750 ml, recycled steel, dishwasher safe, keeps drinks cold for 24 hours).",
        ["Between 40 and 80 words", "Mentions that it is dishwasher safe",
         "Ends with a question addressed to the reader", "Does not use the word 'revolutionary'"],
        on_iteration=show, max_iterations=5)

    print(f"\n{result.stop_reason} after {result.iterations} iteration(s) (loop {result.loop_id})")
    print(result.artifact)


if __name__ == "__main__":
    main()
