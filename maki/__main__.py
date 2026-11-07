"""
Main entry point for the Maki framework CLI.

Usage:
    maki                              Print usage help.
    maki serve --config <file> [options]

serve options:
    --host HOST           Bind address (default: 127.0.0.1)
    --port PORT           Listen port  (default: 8100)
    --api-key KEY         Bearer token for auth (or set MAKI_API_KEY, which keeps it out of `ps`)
    --max-body-mb MB      Largest request body accepted (default: 4)
    --max-concurrency N   Answer 503 beyond N requests in flight (default: unlimited)
    --graceful-timeout S  Seconds in-flight requests get to finish on shutdown (default: 30)
    --tls-cert FILE       TLS certificate file (enables HTTPS)
    --tls-key  FILE       TLS private-key file
"""
import argparse
import os
import sys

from .config import DEFAULT_MODEL, DEFAULT_OLLAMA_BASE_URL
from .logging_config import configure_logging
from .makiLLama import MakiLLama
from .agents import Agent, AgentManager

__all__ = ['MakiLLama', 'Agent', 'AgentManager']


def _is_loopback(host: str) -> bool:
    import ipaddress
    if host in ("localhost", ""):
        return host == "localhost"
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _resolve_api_key(args: argparse.Namespace) -> None:
    """Settle ``args.api_key``: the flag wins, else ``MAKI_API_KEY``. A key on the command line is visible to
    every user on the machine (``ps``) and lands in shell history, so say so."""
    if args.api_key:
        print("Warning: --api-key is visible in the process list; prefer the MAKI_API_KEY environment variable.",
              file=sys.stderr)
    else:
        args.api_key = os.environ.get("MAKI_API_KEY") or None


def _warn_if_exposed(args: argparse.Namespace) -> None:
    """Say loudly when the server is reachable beyond this machine without authentication."""
    if not args.api_key and not _is_loopback(args.host):
        print(f"WARNING: listening on {args.host} with no API key: anyone who can reach this port can run the "
              "agent and its plugins. Set MAKI_API_KEY (or --api-key), or bind to 127.0.0.1 behind a trusted "
              "proxy.", file=sys.stderr)
    if not args.api_key and not args.tls_cert and not _is_loopback(args.host):
        print("WARNING: no TLS either; requests and answers cross the network in clear text.", file=sys.stderr)


def _refuse_open_loops(args: argparse.Namespace, server_cfg) -> None:
    """Exit unless loops are off, or the server is loopback-only or has an API key."""
    if server_cfg.judge is None or args.api_key or _is_loopback(args.host):
        return
    print("Error: loops are enabled (the config has a 'judge:' section) but the server would listen on "
          f"{args.host} without --api-key; anyone who can reach it could start and approve loops. "
          "Pass --api-key, bind to 127.0.0.1, or remove the 'judge:' section.", file=sys.stderr)
    sys.exit(1)


def _cmd_serve(args: argparse.Namespace) -> None:
    if args.tls_key and not args.tls_cert:
        print("Error: --tls-key requires --tls-cert", file=sys.stderr)
        sys.exit(1)

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install 'maki-framework[distributed]'",
              file=sys.stderr)
        sys.exit(1)

    try:
        from .distributed.config_loader import load_server_config
        from .distributed.server import create_app
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        server_cfg = load_server_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    _resolve_api_key(args)
    _refuse_open_loops(args, server_cfg)
    _warn_if_exposed(args)

    max_body_mb = getattr(args, "max_body_mb", 4.0)
    max_concurrency = getattr(args, "max_concurrency", None)
    if max_body_mb <= 0 or (max_concurrency is not None and max_concurrency < 1) \
            or getattr(args, "graceful_timeout", 30.0) <= 0:
        print("Error: --max-body-mb, --max-concurrency and --graceful-timeout must be positive", file=sys.stderr)
        sys.exit(1)

    agent = server_cfg.agent
    app = create_app(agent, api_key=args.api_key, judge=server_cfg.judge, loop_caps=server_cfg.loop_caps,
                     max_body_bytes=int(max_body_mb * 1024 * 1024))

    scheme = "https" if args.tls_cert else "http"
    print(f"Maki Agent Server — {agent.name!r} ({agent.role or 'no role'})")
    print(f"  Listening on  {scheme}://{args.host}:{args.port}")
    print(f"  Auth          {'Bearer token' if args.api_key else 'disabled (trusted network)'}")
    print(f"  TLS           {'enabled' if args.tls_cert else 'disabled'}")
    print(f"  Plugins       {list(agent.plugins.keys()) or 'none'}")
    loops = f"enabled (judge: {server_cfg.judge.name})" if server_cfg.judge else "disabled (no judge: section)"
    print(f"  Loops         {loops}")

    # Without a bound, stopping the server (a deploy, a SIGTERM) waits for the slowest model call.
    uvicorn_kwargs: dict = {"host": args.host, "port": args.port,
                            "timeout_graceful_shutdown": getattr(args, "graceful_timeout", 30.0)}
    if max_concurrency is not None:
        uvicorn_kwargs["limit_concurrency"] = max_concurrency
    if args.tls_cert:
        uvicorn_kwargs["ssl_certfile"] = args.tls_cert
    if args.tls_key:
        uvicorn_kwargs["ssl_keyfile"] = args.tls_key
    uvicorn.run(app, **uvicorn_kwargs)


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
    serve.add_argument("--max-body-mb", type=float, default=4.0, metavar="MB",
                       help="Largest request body accepted, in MB (default: 4)")
    serve.add_argument("--max-concurrency", type=int, default=None, metavar="N",
                       help="Answer 503 when more than N requests are in flight (default: unlimited)")
    serve.add_argument("--graceful-timeout", type=float, default=30.0, metavar="S",
                       help="Seconds in-flight requests get to finish when the server is stopped (default: 30)")
    serve.add_argument("--tls-cert", default=None, metavar="FILE",
                       help="TLS certificate file — enables HTTPS when set")
    serve.add_argument("--tls-key", default=None, metavar="FILE",
                       help="TLS private-key file (required when --tls-cert is set)")

    # maki eval
    evaluate = sub.add_parser("eval", help="Compare models on a built-in or your own eval suite")
    from .eval import add_cli_arguments
    add_cli_arguments(evaluate)

    args = parser.parse_args()
    configure_logging()

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "eval":
        from .eval import run_cli
        raise SystemExit(run_cli(args))
    else:
        parser.print_help()
        print("\nExample:")
        print(f"  {DEFAULT_MODEL!r} via Ollama at {DEFAULT_OLLAMA_BASE_URL}")
        print("  from maki import MakiLLama")
        print("  from maki import Agent")


if __name__ == "__main__":
    main()
