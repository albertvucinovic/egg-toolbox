from __future__ import annotations

import json
from functools import partial
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from .openai import chat_completions
from ..orchestrator import Orchestrator


def create_app(orchestrator: Orchestrator) -> Starlette:
    """Create the ASGI application with all routes."""

    async def handle_chat_completions(request: Request) -> Any:
        return await chat_completions(request, orchestrator)

    async def list_models(request: Request) -> JSONResponse:
        model_name = orchestrator._backend.model_name()
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "local",
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
