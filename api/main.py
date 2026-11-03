import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.services import scheduler_service
from api.routes.corpuses import router as corpuses_router
from api.routes.annotation import router as annotations_router
from api.routes.domains_list import router as domains_list_router
from api.routes.jobs import router as jobs_router
from api.routes.knowledge_graph import router as kg_router
from definitions import ROOT_DIR

JOBS_DIR = os.path.join(ROOT_DIR, 'api', 'jobs')

@asynccontextmanager
async def lifespan(app : FastAPI):

    # os.makedirs(JOBS_DIR, exist_ok=True)
    # scheduler_service.load_from_dir(JOBS_DIR)
    asyncio.create_task(scheduler_service.worker())

    yield

    # scheduler_service.save_jobs_to_dir(JOBS_DIR)

app = FastAPI(lifespan=lifespan, redirect_slashes=False)

app.include_router(corpuses_router)
app.include_router(annotations_router)
app.include_router(domains_list_router)
app.include_router(jobs_router)
app.include_router(kg_router)