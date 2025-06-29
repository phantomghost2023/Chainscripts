from fastapi import FastAPI
from prometheus_client import make_asgi_app

app = FastAPI()
metrics_app = make_asgi_app()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "components": {"healing_executor": True, "config_watcher": True},
    }


app.mount("/metrics", metrics_app)
