import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, status, HTTPException
from starlette.responses import FileResponse
from api.services import scheduler_service
from api.routes.corpuses import router as corpuses_router
from api.routes.annotation import router as annotations_router
from api.routes.domains_list import router as domains_list_router
from api.routes.jobs import router as jobs_router
from api.routes.knowledge_graph import router as kg_router
from api.routes.feature_extraction import router as features_router
from api.routes.graph_learning import router as gl_router
from api.routes.expertise_maps import router as em_router
from api.routes.llms import router as llms_router
from api.routes.graph import router as graph_router
from api.constants import JOBS_DIR, VIS_DIR
from api.db.neo4j import init_driver, get_config, close_driver
from src.manifest import manifest
from src.helpers.torch_utils import get_all_devices
from typing import List

@asynccontextmanager
async def lifespan(app : FastAPI):

    # os.makedirs(JOBS_DIR, exist_ok=True)
    # scheduler_service.load_from_dir(JOBS_DIR)
    asyncio.create_task(scheduler_service.worker())
    await init_driver(get_config())

    yield

    close_driver()
    # scheduler_service.save_jobs_to_dir(JOBS_DIR)

app = FastAPI(lifespan=lifespan, redirect_slashes=False)

@app.get("/available-embedders")
def embedders() -> List[str]:

    def _kind(name : str) -> str:
        return name.split('/')[0].lower()

    opts = {
        "sentence-transformers" : "sbert",
        "qwen" : "qwen",
        "intfloat" : "e5"
    }

    return [
        f"{opts[_kind(hf_model.name)]}:{hf_model.name}"
        for hf_model in manifest.hf_models
        if _kind(hf_model.name) in opts
    ]

@app.get('/download/{filename}')
def download(filename : str) -> FileResponse:

    location = os.path.join(VIS_DIR, filename)

    if not os.path.exists(location):

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "reason" : f"{filename} is not found at : {VIS_DIR}.",
                "filename" : filename
            }
        )

    return FileResponse(
        location, 
        media_type='application/octet-stream',
        filename=filename
    )


@app.get("/devices")
def devices() -> List[str]:
    devices = get_all_devices()
    return [device for device in devices if device != "cuda"]

app.include_router(corpuses_router)
app.include_router(annotations_router)
app.include_router(domains_list_router)
app.include_router(jobs_router)
app.include_router(kg_router)
app.include_router(features_router)
app.include_router(gl_router)
app.include_router(em_router)
app.include_router(llms_router)
app.include_router(graph_router)