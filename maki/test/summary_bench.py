"""
Live benchmark of the summary memory tier against a real Ollama model.

Facts are stated in the first turns, filler turns follow until the conversation no longer fits
``token_budget``, then the agent is asked for the facts. Three setups are compared:

    full      generous budget, nothing evicted (the upper bound)
    drop      small budget, evicted turns are dropped
    summary   the same small budget with summarize_older=True

    python -m maki.test.summary_bench [--model gemma4:26b] [--budget 300] [--filler 10] [--runs 1]

Needs a running Ollama server; prints a table and exits non-zero only on errors.
"""
import argparse
import logging
import time

from maki import init_backend
from maki.agents import Agent
from maki.objects import GenerationConfig, MemoryPolicy

FACTS = {
    "codename": ("The project codename is Bluefin.", "bluefin"),
    "deadline": ("The launch deadline is March 14.", "march 14"),
    "budget": ("The approved budget is 42,000 euros.", "42,000"),
    "owner": ("The project owner is Priya Raman.", "priya"),
}
FILLER = [
    "Give me one tip for writing clear emails.",
    "Name a good habit for staying focused.",
    "Suggest a healthy breakfast idea.",
    "What is one way to learn a language faster?",
    "Recommend a way to organise a small team meeting.",
    "Give me a short tip for better sleep.",
    "Suggest a simple weekend hobby.",
    "What is a good way to review code?",
    "Name one tip for giving presentations.",
    "Suggest a way to keep notes tidy.",
    "Give me a tip for saving time on chores.",
    "Name a simple stretch to do at a desk.",
]
QUESTION = ("Without asking me anything, list the project codename, the launch deadline, the approved "
            "budget and the project owner, in one line.")


def run(setup: str, args) -> dict:
    backend = init_backend(f"ollama:{args.model}", think=False, timeout=300)
    config = GenerationConfig(temperature=0, max_tokens=90)
    policies = {
        "full": MemoryPolicy(token_budget=20000),
        "drop": MemoryPolicy(token_budget=args.budget),
        "summary": MemoryPolicy(token_budget=args.budget, summarize_older=True, summary_chars=args.summary_chars),
    }
    agent = Agent("assistant", backend, stateful=True, memory=policies[setup],
                  instructions="Answer in at most two short sentences.")
    backend.config = config
    started = time.time()
    for statement, _ in FACTS.values():
        agent.execute_task(f"Remember this for later: {statement}")
    for i in range(args.filler):
        agent.execute_task(FILLER[i % len(FILLER)])
    before = agent.usage.snapshot()
    answer = agent.execute_task(QUESTION)
    final = agent.usage - before
    memory = agent._conversation_memory
    found = [name for name, (_, needle) in FACTS.items() if needle in answer.lower()]
    return {
        "setup": setup, "recalled": len(found), "missing": sorted(set(FACTS) - set(found)),
        "calls": agent.usage.calls, "final_prompt_tokens": final.prompt_tokens,
        "turns_kept": len(memory) // 2, "dropped": memory.dropped_pairs, "summarized": memory.summarized_pairs,
        "seconds": round(time.time() - started, 1), "answer": answer.strip().replace("\n", " ")[:110],
        "summary": memory.summary[:160],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--budget", type=int, default=300, help="token_budget for drop and summary")
    parser.add_argument("--summary-chars", type=int, default=400)
    parser.add_argument("--filler", type=int, default=10)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    print(f"model={args.model} budget={args.budget} filler={args.filler} summary_chars={args.summary_chars}\n")
    print(f"{'setup':8} {'recalled':>8} {'calls':>5} {'last prompt tok':>15} {'kept':>4} {'dropped':>7} "
          f"{'summarised':>10} {'sec':>6}")
    for setup in ("full", "drop", "summary"):
        for _ in range(args.runs):
            r = run(setup, args)
            print(f"{r['setup']:8} {str(r['recalled']) + '/4':>8} {r['calls']:>5} {r['final_prompt_tokens']:>15} "
                  f"{r['turns_kept']:>4} {r['dropped']:>7} {r['summarized']:>10} {r['seconds']:>6}")
            print(f"         answer : {r['answer']}")
            if r["summary"]:
                print(f"         summary: {r['summary']}")
            if r["missing"]:
                print(f"         missing: {', '.join(r['missing'])}")


if __name__ == "__main__":
    main()
