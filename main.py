# Uvicorn starts the gateway by importing `main:app`.
# The real application wiring lives in `app_factory.create_app` so tests can
# build the same FastAPI app with test settings and fake upstream clients.
from app_factory import create_app

app = create_app()
