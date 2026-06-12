"""
AgentServer: FastAPI application that exposes a single Maki Agent over HTTP.

Install optional dependencies before use:
    pip install "maki[distributed]"

Endpoints
---------
GET  /health              Liveness check; returns agent_id, name, role.
                          Always unauthenticated (load-balancer/k8s probes).
GET  /info                Agent metadata: plugins, backend class, model.
POST /execute             Run a task; returns result + elapsed time.
GET  /stream              SSE token stream for a task (query param: task).
POST /memory/set          Store a key/value in agent memory.
GET  /memory/{key}        Retrieve a memory value.
DELETE /memory/{key}      Remove a memory key.
GET  /history             Return the task_history deque as a list.
DELETE /history           Clear conversation and task history.

Authentication
--------------
If create_app() receives a non-None api_key, every request except GET /health
must carry:
    Authorization: Bearer <api_key>
Omit api_key to run in open/trusted-network mode.
"""
try:
    from fastapi import Depends, FastAPI, HTTPException, Request, Security
    from fastapi.responses import StreamingResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, field_validator
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError as _e:
    raise ImportError(
        "Distributed server requires optional dependencies. "
        'Install them with: pip install "maki[distributed]"'
    ) from _e

import json
import logging
import secrets
import time
import uuid
from typing import Any, Optional

from ..agents.agent import Agent
from ..exceptions import (
    MakiAPIError,
    MakiError,
    MakiNetworkError,
    MakiTimeoutError,
    MakiValidationError,
)

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


class ExecuteRequest(BaseModel):
    task: str
    context: Optional[dict] = None
    use_plugins: bool = False

    @field_validator("task")
    @classmethod
    def task_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("task must be a non-empty string")
        return v


class MemorySetRequest(BaseModel):
    key: str
    value: Any


def create_app(agent: Agent, api_key: Optional[str] = None) -> FastAPI:
    """
    Build and return a FastAPI application wrapping *agent*.

    Args:
        agent:   The Agent instance to serve.
        api_key: If set, every request must carry this value as a Bearer token.
                 Pass None to run without authentication (trusted network only).
    """
    app = FastAPI(title="Maki Agent Server", version="0.1.0")
    app.state.agent = agent
    app.state.api_key = api_key

    # ------------------------------------------------------------------
    # Trace middleware: attach/propagate X-Maki-Trace-Id on every request
    # ------------------------------------------------------------------

    class _TracingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            trace_id = request.headers.get("X-Maki-Trace-Id") or str(uuid.uuid4())
            request.state.trace_id = trace_id
            logger.debug("[trace=%s] → %s %s", trace_id, request.method, request.url.path)
            response = await call_next(request)
            response.headers["X-Maki-Trace-Id"] = trace_id
            logger.debug("[trace=%s] ← %d", trace_id, response.status_code)
            return response

    app.add_middleware(_TracingMiddleware)

    def _auth(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
    ) -> None:
        key: Optional[str] = request.app.state.api_key
        if key is None:
            return
        if credentials is None or not secrets.compare_digest(
            credentials.credentials.encode("utf-8"), key.encode("utf-8")
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")

    # ------------------------------------------------------------------
    # Health / info
    # ------------------------------------------------------------------

    # /health is deliberately unauthenticated: load-balancer and k8s probes
    # cannot send credentials, and the body contains no sensitive data.
    @app.get("/health")
    def health(request: Request):
        ag: Agent = request.app.state.agent
        return {
            "status": "ok",
            "agent_id": ag.agent_id,
            "name": ag.name,
            "role": ag.role,
        }

    @app.get("/info")
    def info(request: Request, _: None = Depends(_auth)):
        ag: Agent = request.app.state.agent
        return {
            "agent_id": ag.agent_id,
            "name": ag.name,
            "role": ag.role,
            "plugins": list(ag.plugins.keys()),
            "backend": type(ag.maki).__name__,
            "model": getattr(ag.maki, "model", None),
        }

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    @app.post("/execute")
    def execute(req: ExecuteRequest, request: Request, _: None = Depends(_auth)):
        ag: Agent = request.app.state.agent
        trace_id: str = getattr(request.state, "trace_id", str(uuid.uuid4()))
        t0 = time.time()
        logger.debug("[trace=%s] executing task for agent '%s'", trace_id, ag.name)
        # Client input errors echo the validation message (it describes the
        # caller's own request); everything else returns a generic body and
        # is logged in full server-side only, so internal URLs/paths never
        # reach remote callers.
        try:
            result = ag.execute_task(req.task, req.context, req.use_plugins)
        except (MakiValidationError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except MakiTimeoutError as e:
            logger.error("[trace=%s] backend timeout: %s", trace_id, e)
            raise HTTPException(status_code=504, detail="Agent backend timed out")
        except MakiNetworkError as e:
            logger.error("[trace=%s] backend unreachable: %s", trace_id, e)
            raise HTTPException(status_code=502, detail="Agent backend unavailable")
        except MakiAPIError as e:
            logger.error("[trace=%s] backend API error: %s", trace_id, e)
            raise HTTPException(status_code=400, detail="Agent backend rejected the request")
        except MakiError:
            logger.exception("[trace=%s] agent error", trace_id)
            raise HTTPException(status_code=500, detail="Internal agent error")
        except Exception:
            logger.exception("[trace=%s] unexpected error", trace_id)
            raise HTTPException(status_code=500, detail="Internal server error")
        elapsed = round(time.time() - t0, 3)
        logger.debug("[trace=%s] task completed in %.3fs", trace_id, elapsed)
        return {
            "result": result,
            "agent_id": ag.agent_id,
            "elapsed": elapsed,
            "trace_id": trace_id,
        }

    @app.get("/stream")
    def stream(task: str, request: Request, use_plugins: bool = False, _: None = Depends(_auth)):
        ag: Agent = request.app.state.agent

        def _sse():
            try:
                for chunk in ag.stream_task(task, use_plugins=use_plugins):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            except Exception:
                logger.exception("stream failed for agent '%s'", ag.name)
                yield f"data: {json.dumps({'error': 'stream failed; see server logs'})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    @app.post("/memory/set")
    def memory_set(req: MemorySetRequest, request: Request, _: None = Depends(_auth)):
        request.app.state.agent.remember(req.key, req.value)
        return {"ok": True}

    @app.get("/memory")
    def memory_list(request: Request, _: None = Depends(_auth)):
        return {"memory": dict(request.app.state.agent.memory)}

    @app.delete("/memory")
    def memory_clear(request: Request, _: None = Depends(_auth)):
        request.app.state.agent.clear_memory()
        return {"ok": True}

    @app.get("/memory/{key}")
    def memory_get(key: str, request: Request, _: None = Depends(_auth)):
        ag: Agent = request.app.state.agent
        if key not in ag.memory:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found")
        return {"key": key, "value": ag.memory[key]}

    @app.delete("/memory/{key}")
    def memory_delete(key: str, request: Request, _: None = Depends(_auth)):
        ag: Agent = request.app.state.agent
        if key not in ag.memory:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found")
        del ag.memory[key]
        return {"ok": True}

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    @app.get("/history")
    def history_list(request: Request, _: None = Depends(_auth)):
        ag: Agent = request.app.state.agent
        return {"history": list(ag.task_history)}

    @app.delete("/history")
    def history_clear(request: Request, _: None = Depends(_auth)):
        ag: Agent = request.app.state.agent
        ag.reset_conversation()
        ag.task_history.clear()
        return {"ok": True}

    return app
