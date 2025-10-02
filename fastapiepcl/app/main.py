from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    """Load .env file if it exists"""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.split('#')[0].strip()  # Remove inline comments
                    os.environ[key] = value
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print(f"⚠️  No .env file found at {env_path}")

# Load .env on startup
load_env_file()

# Routers
from .routers import (
    workbooks,
    wordclouds,
    maps,
    analytics_general,
    analytics_conversion,
    analytics_advanced,
    analytics_predictive,
    filters,
    data_health,
    agent,
    agent_ws,  # WebSocket for ultra-fast streaming
    data,
)


def create_app() -> FastAPI:
    app = FastAPI(title="Safety Co-pilot API", version="0.1.0")

    # CORS (explicit local origins + regex; adjust for production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "*"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Include feature routers
    app.include_router(workbooks.router)
    app.include_router(wordclouds.router)
    app.include_router(maps.router)
    app.include_router(analytics_general.router)
    app.include_router(analytics_conversion.router)
    app.include_router(analytics_advanced.router)
    app.include_router(analytics_predictive.router)
    app.include_router(filters.router)
    app.include_router(data_health.router)
    app.include_router(agent.router)
    app.include_router(agent_ws.router)  # WebSocket endpoints
    app.include_router(data.router)

    return app


app = create_app()

