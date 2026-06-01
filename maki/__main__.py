"""
Main entry point for the Maki framework CLI.

Usage:
    maki                   Print usage help.
    maki serve --config <file> [--host HOST] [--port PORT] [--api-key KEY]
"""
import argparse
import logging
import sys

from .config import (
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
)

from .makiLLama import MakiLLama
from .agents import Agent, AgentManager

__all__ = ['MakiLLama', 'Agent', 'AgentManager']


def configure_logging():
    logging.basicConfig(
        level=DEFAULT_LOG_LEVEL,
        format=DEFAULT_LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install 'maki[distributed]'",
              file=sys.stderr)
        sys.exit(1)

    try:
        from .distributed.config_loader import load_agent_from_config
        from .distributed.server import create_app
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        agent = load_agent_from_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    app = create_app(agent, api_key=args.api_key)

    print(f"Maki Agent Server — {agent.name!r} ({agent.role or 'no role'})")
    print(f"  Listening on  http://{args.host}:{args.port}")
    print(f"  Auth          {'Bearer token' if args.api_key else 'disabled (trusted network)'}")
    print(f"  Plugins       {list(agent.plugins.keys()) or 'none'}")

    uvicorn.run(app, host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="maki",
        description="Maki framework — multi-agent LLM interactions",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # maki serve
    serve = sub.add_parser("serve", help="Serve an agent over HTTP")
    serve.add_argument("--config", required=True, metavar="FILE",
                       help="Path to agent YAML config file")
    serve.add_argument("--host", default="127.0.0.1",
                       help="Bind host (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8100,
                       help="Listen port (default: 8100)")
    serve.add_argument("--api-key", default=None, metavar="KEY",
                       help="Bearer token for authentication (omit for open access)")

    args = parser.parse_args()

    if args.command == "serve":
        _cmd_serve(args)
    else:
        parser.print_help()
        print(f"\nExample:")
        print(f"  {DEFAULT_MODEL!r} via Ollama at {DEFAULT_OLLAMA_BASE_URL}")
        print(f"  from maki import MakiLLama")
        print(f"  from maki import Agent")


if __name__ == "__main__":
    main()
