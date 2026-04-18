from __future__ import annotations

import json
import time
import traceback

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from .openai import chat_completions
from ..orchestrator import Orchestrator


def _error_response(status: int, message: str, error_type: str, code: str | None = None) -> JSONResponse:
    """Return an OpenAI-format error response."""
    error: dict = {
        "message": message,
        "type": error_type,
    }
    if code is not None:
        error["code"] = code
    return JSONResponse({"error": error}, status_code=status)


def create_app(orchestrator: Orchestrator) -> Starlette:
    """Create the ASGI application with all routes."""

    async def handle_chat_completions(request: Request) -> JSONResponse:
        # Parse JSON body
        try:
            raw = await request.body()
            body = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            return _error_response(
                400,
                f"Invalid JSON in request body: {e}",
                "invalid_request_error",
                "invalid_json",
            )

        # Validate messages field
        if "messages" not in body:
            return _error_response(
                400,
                "Missing required field: 'messages'",
                "invalid_request_error",
                "missing_field",
            )
        if not isinstance(body["messages"], list) or len(body["messages"]) == 0:
            return _error_response(
                400,
                "'messages' must be a non-empty array",
                "invalid_request_error",
                "invalid_field",
            )

        # Dispatch to handler
        try:
            return await chat_completions(body, orchestrator)
        except Exception:
            traceback.print_exc()
            return _error_response(
                500,
                "Internal server error",
                "server_error",
                "internal_error",
            )

    async def list_models(request: Request) -> JSONResponse:
        model_name = orchestrator._backend.model_name()
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "egg-toolbox",
            }],
        })

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    routes = [
        Route("/v1/chat/completions", handle_chat_completions, methods=["POST"]),
        Route("/v1/models", list_models, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
    ]

    return Starlette(
        routes=routes,
        middleware=[
            Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
        ],
    )
