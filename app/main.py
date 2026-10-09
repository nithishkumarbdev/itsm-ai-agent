import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app import models  # noqa: F401  (importing registers the tables on Base)
from app.api.routes import health, tickets
from app.core.config import settings
from app.db.database import Base, engine
from app.lifecycle import InvalidTransition

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Shortcut: create missing tables on startup. Fine at this size; once existing
    # tables start changing we should switch to migrations (Alembic).
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(tickets.router)


# --- Error handling for every endpoint -------------------------------------------
# Details go to the log; the client only gets a generic message, so no connection
# strings, hostnames or SQL leak out. If a request fails, its transaction is rolled
# back when the session closes, so nothing half-done is saved.


@app.exception_handler(InvalidTransition)
async def invalid_transition_handler(_request: Request, exc: InvalidTransition):
    return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})


@app.exception_handler(OperationalError)  # cannot connect / connection lost
async def database_unavailable_handler(request: Request, exc: OperationalError):
    logger.error("Database unavailable: %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "The database is currently unavailable. Try again later."},
    )


@app.exception_handler(SQLAlchemyError)  # any other database error
async def database_error_handler(request: Request, exc: SQLAlchemyError):
    logger.error("Database error: %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "The request could not be completed due to an internal error."},
    )
