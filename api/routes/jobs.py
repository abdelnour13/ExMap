import os
from fastapi import APIRouter, Path, Query, HTTPException, status, Header
from fastapi.sse import EventSourceResponse
from typing import Optional, Annotated, AsyncIterable, List
from api.services.scheduler import JobInfo, Job, JobType, Event, STATUS
from api.services import scheduler_service, kg_construction_service, annotation_serivce, corpuses_service
from src.commands.data_download.main import Config as DataDownloaderConfig, DataDownloader
from api.wrappers import annotation, knowledge_graph_construction

router = APIRouter(prefix="/jobs")

@router.post("/kg_construction/{kg_id}/start")
async def start_kg_construction(
    kg_id : str = Path(),
    x_api_key : Annotated[str | None, Header()] = None
) -> JobInfo:
    
    if kg_construction_service.get_by_id(kg_id) is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={ "id" : kg_id }
        )

    if x_api_key is None:
    
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={ "reason" : "Header api_key is not set." }
        )

    config = knowledge_graph_construction.Config(
        graph_id=kg_id,
        stage=None
    )

    downloader = knowledge_graph_construction.Pipeline.from_config(config=config)

    job = Job(
        log_filename=config.log_filename,
        cmd=downloader,
        type="KG_CONSTRUCTION",
        config_id=kg_id,
        conf=config
    )

    added = await scheduler_service.add_job(job)

    if not added:

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={ "id" : kg_id, "reason" : f"another job that operates on '{kg_id}' is pending or running." }
        )

    return job.info

@router.post("/download/{corpus_id}/start")
async def start_data_download(corpus_id : str = Path()) -> JobInfo:
    
    if corpuses_service.get_by_id(corpus_id) is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={ "id" : corpus_id }
        )

    config = DataDownloaderConfig(
        log=True,
        corpus_id=corpus_id,
        resume=False,
        cache=False
    )

    downloader = DataDownloader.from_config(config=config)

    job = Job(
        log_filename=config.log_filename,
        cmd=downloader,
        type="DOWNLOAD",
        config_id=corpus_id,
        conf=config
    )

    added = await scheduler_service.add_job(job)

    if not added:

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={ "corpus_id" : corpus_id, "reason" : f"another job that operates on '{corpus_id}' is pending or running." }
        )

    return job.info

@router.post("/annotation/{id}/start")
async def start_data_annotation(
    id : str = Path(),
    resume : bool = Query(default=True),
    x_api_key : Annotated[str | None, Header()] = None
) -> JobInfo:
    
    if annotation_serivce.get_by_id(id) is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={ "id" : id }
        )

    if x_api_key is None:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={ "reason" : "Header api_key is not set." }
        )

    config = annotation.Config(
        log=True,
        resume=resume,
        cache=True,
        annotation_id=id,
        api_key=x_api_key
    )

    cmd = annotation.CorpusAnnotatator.from_config(config=config)

    job = Job(
        log_filename=config.log_filename,
        cmd=cmd,
        type="ANNOTATION",
        config_id=id,
        conf=config
    )

    added = await scheduler_service.add_job(job)

    if not added:

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={ "id" : id, "reason" : f"another job that operates on '{id}' is pending or running." }
        )

    return job.info

@router.post("/{job_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_job(job_id : str = Path()) -> JobInfo:

    job = scheduler_service.get_job(job_id)
    
    if job is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={ "job_id" : job_id }
        )

    if job.status in ("COMPLETED", "ERROR"):

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "job_id" : job_id,
                "reason" : f"Cannot cancel job: job is already {job.status}"
            }
        )

    scheduler_service.cancel_job(job.id)

    return job.info

@router.get("/{job_id}")
def get_job(job_id : str = Path()) -> JobInfo:

    job = scheduler_service.get_job(id=job_id)

    if job is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={ "job_id" : job_id }
        )

    return job.info

@router.get("")
def get_jobs(
    status : Optional[STATUS] = Query(default=None),
    type : Optional[JobType] = Query(default=None)
) -> List[JobInfo]:
    jobs = scheduler_service.get_all_jobs(status=status, type=type)
    return [job.info for job in jobs]

@router.get("/sse/{job_id}", response_class=EventSourceResponse)
async def stream_job_events(
    job_id : str = Path(),
    last_event_id: Annotated[int | None, Header()] = None
) -> AsyncIterable[Event]:

    job = scheduler_service.get_job(id=job_id)

    if job is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={ "job_id" : job_id }
        )

    last_event_id = (last_event_id + 1) if last_event_id is not None else 0   

    async for event in scheduler_service.stream_events(id=job_id, start=last_event_id):
        yield event