"""
AgentServer: FastAPI application that exposes a single Maki Agent over HTTP.

Install optional dependencies before use:
    pip install "maki-framework[distributed]"

Endpoints
---------
GET  /health              Liveness check; returns agent_id, name, role.
                          Always unauthenticated (load-balancer/k8s probes).
GET  /info                Agent metadata: plugins, backend class, model.
POST /execute             Run a task; returns result + elapsed time.
POST /stream              SSE token stream for a task (same body as /execute;
                          POST keeps task text out of access logs, §2.5).
POST /loop                Start a background loop: iterate the agent until a judge
                          confirms every (text) requirement. 202 + loop id. Needs a
                          ``judge:`` section in the config, else 503.
GET  /loop                List loops (running, finished, and checkpoints on disk).
GET  /loop/{id}           Loop status.
GET  /loop/{id}/events    SSE stream of loop events (replay with Last-Event-ID or ?after=N).
DELETE /loop/{id}         Cancel a running loop (takes effect between iterations).
POST /loop/{id}/resume    Continue a stopped or interrupted loop from its checkpoint.
POST /loop/{id}/approval  Answer a suspended loop's pending approval ({approve, note}); then resume.
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
    from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, Security
    from fastapi.responses import StreamingResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, ConfigDict, field_validator
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError as _e:
    raise ImportError(
        "Distributed server requires optional dependencies. "
        'Install them with: pip install "maki-framework[distributed]"'
    ) from _e

import asyncio
import json
import logging
import secrets
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from ..agents.agent import Agent
from ..exceptions import (
    MakiAPIError,
    MakiError,
    MakiNetworkError,
    MakiTimeoutError,
    MakiValidationError,
)
from .loop_jobs import LOOP_ID_PATTERN, LoopCaps, LoopJobBusy, LoopJobNotFound, LoopJobs, NoPendingApproval

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

_LOOP_BODY_LIMIT = 256 * 1024  # bytes; a loop request is a goal plus at most 30 short requirements
_SSE_POLL_SECONDS = 0.25


class _BodyLimitMiddleware:
    """Answer 413 to a request under ``prefix`` whose body is larger than ``max_bytes``.

    Checks the declared Content-Length first, then counts the bytes actually received, so a
    missing or false Content-Length (chunked uploads) cannot get around it.
    """

    def __init__(self, app, max_bytes: int, prefix: str):
        self.app, self.max_bytes, self.prefix = app, max_bytes, prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith(self.prefix):
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return
        guard = _BodyGuard(self, receive, send)
        try:
            await self.app(scope, guard.receive, guard.send)
        except _TooLarge:
            pass
        if guard.exceeded and not guard.replied:
            await self._reject(send)

    async def _reject(self, send):
        body = json.dumps({"detail": f"Request body larger than {self.max_bytes} bytes"}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


class _TooLarge(Exception):
    pass


class _BodyGuard:
    """Per-request state of :class:`_BodyLimitMiddleware`: counts the body, swaps the framework's
    own 400 for a 413 once the limit is passed."""

    def __init__(self, middleware: "_BodyLimitMiddleware", receive, send):
        self.middleware, self._receive, self._send = middleware, receive, send
        self.received, self.exceeded, self.replied = 0, False, False

    async def receive(self):
        message = await self._receive()
        if message["type"] == "http.request":
            self.received += len(message.get("body", b""))
            if self.received > self.middleware.max_bytes:
                self.exceeded = True
                raise _TooLarge()
        return message

    async def send(self, message):
        if not self.exceeded:
            await self._send(message)
        elif not self.replied:  # the framework turns the read failure into its own 400; answer 413
            self.replied = True
            await self.middleware._reject(self._send)


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


class LoopRequest(BaseModel):
    """Start a loop. Text requirements only; unknown fields (command, cwd, loop_dir, ...) are rejected."""
    model_config = ConfigDict(extra="forbid")

    goal: str
    requirements: List[Union[str, Dict[str, Any]]]
    max_iterations: Optional[int] = None
    max_seconds: Optional[float] = None
    stall_limit: Optional[int] = None
    context: Optional[Literal["shared", "fresh"]] = None
    use_plugins: bool = False
    max_tokens: Optional[int] = None
    require_approval: bool = False


class LoopApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approve: bool
    note: str = ""


class LoopResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iterations: Optional[int] = None
    max_seconds: Optional[float] = None
    max_tokens: Optional[int] = None


class MemorySetRequest(BaseModel):
    key: str
    value: Any


router = APIRouter()


# ----------------------------------------------------------------------
# Trace middleware: attach/propagate X-Maki-Trace-Id on every request
# ----------------------------------------------------------------------


class _TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("X-Maki-Trace-Id") or str(uuid.uuid4())
        request.state.trace_id = trace_id
        logger.debug("[trace=%s] → %s %s", trace_id, request.method, request.url.path)
        response = await call_next(request)
        response.headers["X-Maki-Trace-Id"] = trace_id
        logger.debug("[trace=%s] ← %d", trace_id, response.status_code)
        return response


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
@router.get("/health")
def health(request: Request):
    ag: Agent = request.app.state.agent
    return {
        "status": "ok",
        "agent_id": ag.agent_id,
        "name": ag.name,
        "role": ag.role,
    }


@router.get("/info")
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


@router.post("/execute")
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


@router.post("/stream")
def stream(req: ExecuteRequest, request: Request, _: None = Depends(_auth)):
    ag: Agent = request.app.state.agent

    def _sse():
        try:
            for chunk in ag.stream_task(req.task, req.context, use_plugins=req.use_plugins):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except Exception:
            logger.exception("stream failed for agent '%s'", ag.name)
            yield f"data: {json.dumps({'error': 'stream failed; see server logs'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


# ------------------------------------------------------------------
# Loops
# ------------------------------------------------------------------


def _jobs(request: Request) -> LoopJobs:
    jobs: Optional[LoopJobs] = request.app.state.loop_jobs
    if jobs is None:
        raise HTTPException(status_code=503, detail="Loops are disabled: no judge is configured")
    return jobs


def _loop_id(loop_id: str) -> str:
    if not LOOP_ID_PATTERN.match(loop_id):
        raise HTTPException(status_code=404, detail="Loop not found")
    return loop_id


@router.post("/loop", status_code=202)
def loop_start(req: LoopRequest, request: Request, _: None = Depends(_auth)):
    jobs = _jobs(request)
    try:
        job = jobs.start(req.goal, req.requirements, max_iterations=req.max_iterations,
                         max_seconds=req.max_seconds, stall_limit=req.stall_limit,
                         context=req.context, use_plugins=req.use_plugins, max_tokens=req.max_tokens,
                         require_approval=req.require_approval)
    except LoopJobBusy as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return job.snapshot()


@router.get("/loop")
def loop_list(request: Request, _: None = Depends(_auth)):
    return {"loops": _jobs(request).list()}


@router.get("/loop/{loop_id}")
def loop_status(loop_id: str, request: Request, _: None = Depends(_auth)):
    try:
        return _jobs(request).snapshot(_loop_id(loop_id))
    except LoopJobNotFound:
        raise HTTPException(status_code=404, detail="Loop not found")


def _stored_stream(snap: dict, start: int):
    """Events of a loop this process no longer holds: one synthetic ``stopped`` event from its checkpoint."""
    stopped = {"type": "stopped", "stop_reason": snap["stop_reason"] or snap["status"],
               "iterations": snap["iterations"], "tokens": snap["tokens"], "detail": snap["detail"],
               "error": snap["error"], "artifact": snap["artifact"]}
    if start == 0:
        yield f"id: 0\ndata: {json.dumps(stopped)}\n\n"
    yield "data: [DONE]\n\n"


async def _live_stream(job, start: int, heartbeat: float):
    """Events of a loop held in memory. Polls instead of blocking a worker thread per watcher, and
    a dropped client cancels this coroutine at the sleep."""
    index, quiet_since = start, time.monotonic()
    while True:
        events, finished = job.events_from(index)
        for event in events:
            yield f"id: {index}\ndata: {json.dumps(event)}\n\n"
            index += 1
            quiet_since = time.monotonic()
        if finished and not events:
            break
        if not events:
            if time.monotonic() - quiet_since >= heartbeat:
                yield ": keep-alive\n\n"
                quiet_since = time.monotonic()
            await asyncio.sleep(min(_SSE_POLL_SECONDS, heartbeat))
    yield "data: [DONE]\n\n"


def _first_event_index(after: Optional[int], last_event_id: Optional[str]) -> int:
    if after is not None:
        return after + 1
    if last_event_id is not None and last_event_id.strip().lstrip("-").isdigit():
        return max(int(last_event_id) + 1, 0)
    return 0


@router.get("/loop/{loop_id}/events")
def loop_events(loop_id: str, request: Request, after: Optional[int] = Query(None, ge=-1),
                last_event_id: Optional[str] = Header(None), _: None = Depends(_auth)):
    jobs = _jobs(request)
    loop_id = _loop_id(loop_id)
    start = _first_event_index(after, last_event_id)
    job = jobs.get(loop_id)
    if job is not None:
        stream = _live_stream(job, start, request.app.state.loop_heartbeat_seconds)
    else:
        try:
            stream = _stored_stream(jobs.snapshot(loop_id), start)
        except LoopJobNotFound:
            raise HTTPException(status_code=404, detail="Loop not found")
    return StreamingResponse(stream, media_type="text/event-stream")


@router.delete("/loop/{loop_id}", status_code=202)
def loop_cancel(loop_id: str, request: Request, _: None = Depends(_auth)):
    jobs = _jobs(request)
    loop_id = _loop_id(loop_id)
    if jobs.cancel(loop_id):
        return {"ok": True, "cancelling": True}
    try:
        jobs.snapshot(loop_id)
    except LoopJobNotFound:
        raise HTTPException(status_code=404, detail="Loop not found")
    raise HTTPException(status_code=409, detail="Loop is not running")


@router.post("/loop/{loop_id}/resume", status_code=202)
def loop_resume(loop_id: str, request: Request, req: Optional[LoopResumeRequest] = None,
                _: None = Depends(_auth)):
    jobs = _jobs(request)
    loop_id = _loop_id(loop_id)
    req = req or LoopResumeRequest()
    try:
        job = jobs.resume(loop_id, max_iterations=req.max_iterations, max_seconds=req.max_seconds,
                          max_tokens=req.max_tokens)
    except LoopJobNotFound:
        raise HTTPException(status_code=404, detail="Loop not found")
    except LoopJobBusy as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return job.snapshot()


@router.post("/loop/{loop_id}/approval")
def loop_approval(loop_id: str, req: LoopApprovalRequest, request: Request, _: None = Depends(_auth)):
    jobs = _jobs(request)
    loop_id = _loop_id(loop_id)
    try:
        return jobs.approve(loop_id, req.approve, req.note)
    except LoopJobNotFound:
        raise HTTPException(status_code=404, detail="Loop not found")
    except NoPendingApproval:
        raise HTTPException(status_code=409, detail="Loop has no pending approval")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------------------------------------------------------
# Memory
# ------------------------------------------------------------------


@router.post("/memory/set")
def memory_set(req: MemorySetRequest, request: Request, _: None = Depends(_auth)):
    request.app.state.agent.remember(req.key, req.value)
    return {"ok": True}


@router.get("/memory")
def memory_list(request: Request, _: None = Depends(_auth)):
    return {"memory": dict(request.app.state.agent.memory)}


@router.delete("/memory")
def memory_clear(request: Request, _: None = Depends(_auth)):
    request.app.state.agent.clear_memory()
    return {"ok": True}


@router.get("/memory/{key}")
def memory_get(key: str, request: Request, _: None = Depends(_auth)):
    ag: Agent = request.app.state.agent
    if key not in ag.memory:
        raise HTTPException(status_code=404, detail=f"Key '{key}' not found")
    return {"key": key, "value": ag.memory[key]}


@router.delete("/memory/{key}")
def memory_delete(key: str, request: Request, _: None = Depends(_auth)):
    ag: Agent = request.app.state.agent
    if key not in ag.memory:
        raise HTTPException(status_code=404, detail=f"Key '{key}' not found")
    del ag.memory[key]
    return {"ok": True}


# ------------------------------------------------------------------
# History
# ------------------------------------------------------------------


@router.get("/history")
def history_list(request: Request, _: None = Depends(_auth)):
    ag: Agent = request.app.state.agent
    return {"history": list(ag.task_history)}


@router.delete("/history")
def history_clear(request: Request, _: None = Depends(_auth)):
    ag: Agent = request.app.state.agent
    ag.reset_conversation()
    ag.task_history.clear()
    return {"ok": True}


def create_app(agent: Agent, api_key: Optional[str] = None, judge: Optional[Agent] = None,
               loop_caps: Optional[LoopCaps] = None) -> FastAPI:
    """
    Build and return a FastAPI application wrapping *agent*.

    Args:
        agent:     The Agent instance to serve.
        api_key:   If set, every request must carry this value as a Bearer token.
                   Pass None to run without authentication (trusted network only).
        judge:     A separate, stateless reviewer agent. Enables the ``/loop`` endpoints;
                   without it they answer 503 (the served agent never judges its own work).
        loop_caps: Operator limits for remote loops (iterations, wall-clock, storage).
    """
    app = FastAPI(title="Maki Agent Server", version="0.1.0")
    app.state.agent = agent
    app.state.api_key = api_key
    app.state.loop_jobs = LoopJobs(agent, judge, loop_caps) if judge is not None else None
    app.state.loop_heartbeat_seconds = 15.0

    app.add_middleware(_TracingMiddleware)
    app.add_middleware(_BodyLimitMiddleware, max_bytes=_LOOP_BODY_LIMIT, prefix="/loop")
    app.include_router(router)
    return app
