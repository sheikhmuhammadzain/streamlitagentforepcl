from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers
from .routers import workbooks, wordclouds, maps, analytics_general, analytics_conversion, agent, data


def create_app() -> FastAPI:
    app = FastAPI(title="Safety Co-pilot API", version="0.1.0")

    # CORS (explicit local origins + regex; adjust for production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\\d+)?$",
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
    app.include_router(agent.router)
    app.include_router(data.router)

    return app


app = create_app()

