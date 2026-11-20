import os
import json
import uuid
import datetime
import asyncio
from src.shared.command_tools import BaseCommand
from src.shared.handlers.base import Handler, Event as _Event
from src.shared.handlers import ConsoleLogger
from src.helpers import load_json_file, save_json_file
from pydantic import BaseModel
from typing import List, Optional, Dict, Literal, AsyncIterable, Set, Type, Tuple
from dataclasses import dataclass, field, asdict
from src.commands.data_download import main as DataDownload
from api.wrappers import annotation as DataAnnotation, knowledge_graph_construction as KGConstruction
from definitions import CORPUS_DIR, ANNOTATION_DIR, KGS_DIR

STATUS = Literal["PENDING", "RUNNING", "ERROR", "COMPLETED"]
JobType = Literal["DOWNLOAD", "ANNOTATION", "KG_CONSTRUCTION", "FEATURE_EXTRACTION", "GRAPH_LEARNING", "MAP_CONSTRUCTION", "NEO4J_UPLOAD"]
LEVEL = Literal["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]

JOB2CLS :Dict[JobType, Tuple[Type[BaseModel], Type[BaseCommand], str]] = {
    "DOWNLOAD" : (DataDownload.Config, DataDownload.DataDownloader, CORPUS_DIR),
    "ANNOTATION" : (DataAnnotation.Config, DataDownload.DataDownloader, ANNOTATION_DIR),
    "KG_CONSTRUCTION" : (KGConstruction.Config, KGConstruction.Pipeline, KGS_DIR)
}

class Event(BaseModel):
    task : str = "main"
    progress : Optional[float] = None
    timestamp : str = field(default_factory=lambda : datetime.now().strftime("%Y-%m-%d %H:%M:%S")) # Timestamp
    level : LEVEL = "INFO" # Level
    message : Optional[str] = None # Message
    extra : Dict = field(default_factory=dict)

class JobInfo(BaseModel):
    id : str
    status : STATUS
    type : JobType
    config_id : str
    last_event : Optional[Event] = None
    log_filename : str

@dataclass
class Job:
    log_filename : str
    type : JobType
    config_id : str
    cmd : BaseCommand
    conf : BaseModel
    id : str = field(default_factory=lambda : str(uuid.uuid4()))
    status : STATUS = "PENDING"
    last_event : Optional[Event] = None
    num_events : int = field(default=0, init=False)

    def __post_init__(self):

        if not os.path.exists(self.log_filename):
            self.num_events = 0
            return

        ### Load logfile initially from logfile then 
        ### increment as more events arrive
        with open(self.log_filename, "r") as f:

            for line in f.readlines():
                self.num_events += 1

    def save_to_file(self, path : str) -> None:

        save_json_file(path, {
            **self.info.model_dump(),
            "conf" : self.conf.model_dump()
        })

    @staticmethod
    def load_from_file(path : str) -> 'Job':

        job_data = load_json_file(path)
        job_data['last_event'] = None if job_data['last_event'] is None else Event(**job_data['last_event'])
        config_cls, cmd_cls, conf_dir = JOB2CLS[job_data['type']]
        job_data['conf'] = config_cls(**job_data['conf'])
        job_data['cmd'] = cmd_cls.from_config(job_data['conf'])

        return Job(**job_data)

    @property
    def info(self) -> JobInfo:

        return JobInfo(
            id=self.id, 
            status=self.status, 
            type=self.type,
            config_id=self.config_id,
            log_filename=self.log_filename,
            last_event=self.last_event,
        )

    def run(self):
        handler = JobHandler(job=self)
        self.cmd.add_handler(handler)
        self.cmd.remove_handlers(cls=ConsoleLogger)
        self.status = "RUNNING"
        self.cmd.run()

class JobHandler(Handler):

    def __init__(self, job : Job):
        super().__init__()
        self.job = job

    def consume(self, event : _Event) -> Event:

        event = Event(**{
            **asdict(event),
            "level" : event.level.name
        })

        self.job.last_event = event
        self.job.num_events += 1

class Scheduler:

    def __init__(self, n : int = 1):

        self.mutix = asyncio.Lock()
        self.job_semaphore = asyncio.Semaphore(n)
        self.queue : asyncio.Queue[Job] = asyncio.Queue() 
        self.jobs : Dict[str, Job] = {} # TO-DO change to more efficient options
        self.running = True

    async def add_job(self, job : Job) -> bool:

        added = False

        async with self.mutix:

            jobs = self.get_all_jobs(status={ "RUNNING", "PENDING" }, type=job.type)
            exists = job.config_id in [j.config_id for j in jobs]

            if not exists:
                await self.queue.put(job)
                self.jobs[job.id] = job
                added = True

        return added

    def get_job(self, id : str) -> Optional[Job]:
        return self.jobs.get(id)

    def get_job_or_throw(self, id : str) -> Job:
        job = self.jobs.get(id)
        assert job is not None, f"job with id {id} doesn't exist."
        return job

    def job_status_from_events(self, job : Job) -> STATUS:

        last_event = job.last_event

        if (last_event is not None) or (last_event.level != "CRITICAL"):
            return "COMPLETED"

        return "ERROR"

    def get_all_jobs(self, 
        status : Optional[STATUS | Set[STATUS]] = None,
        type : Optional[JobType | Set[JobType]] = None
    ) -> List[Job]:

        if status is not None:
            status = { status } if isinstance(status, str) else set(status)

        if type is not None:
            type = { type } if isinstance(type, str) else set(type)

        return [
            job for job in self.jobs.values()
            if ((status is None) or (job.status in status)) and ((type is None) or (job.type in type))
        ]

    def load_from_dir(self, dirname : str) -> None:

        for filename in os.listdir(dirname):

            path = os.path.join(dirname, filename)
            job = Job.load_from_file(path)

            if job.status in ["PENDING", "RUNNING"]:
                self.queue.put(job)

    def save_jobs_to_dir(self, dirname : str) -> None:

        for job in self.jobs.values():
            path = os.path.join(dirname, f'{job.id}.json')
            job.save_to_file(path)
            
    def cancel_job(self, id : str) -> None:
        job = self.get_job_or_throw(id=id)
        job.cmd.cancel()

    def cancel_all(self) -> None:

        self.running = False

        jobs = self.get_all_jobs(status={ "RUNNING" })

        for job in jobs:
            job.cmd.cancel()

    # async def stream_events(self, id : str, start : int = 0) -> AsyncIterable[Event]:

    #     job = self.get_job_or_throw(id=id)

    #     while job.status == "PENDING":
    #         await asyncio.sleep(0.1)

    #     cursor = start

    #     while True:

    #         N = len(job.events)

    #         if (cursor >= N):

    #             if job.status != "RUNNING":
    #                 break
    #             else:
    #                 await asyncio.sleep(0.1)
    #                 continue

    #         local_copy = job.events[cursor:].copy()
    #         cursor += len(local_copy)

    #         for event in local_copy:
    #             yield event

    #         await asyncio.sleep(0.1)

    async def stream_events(self, id : str, start : int = 0) -> AsyncIterable[Event]:
    
        job = self.get_job_or_throw(id=id)
        index = 0

        while job.status == "PENDING":
            await asyncio.sleep(0.1)

        while not os.path.exists(job.log_filename):
            await asyncio.sleep(0.1)

        with open(job.log_filename, "r") as f:

            while True:

                if index >= job.num_events:
                
                    if job.status != "RUNNING":
                        break
                    else:
                        await asyncio.sleep(0.1)
                        continue

                if index < start:
                    index += 1
                    continue

                line = f.readline().strip('\n')
                event = json.loads(s=line)
                event = Event(**event)

                index += 1
                yield event


    async def worker(self) -> None:

        while self.running:

            job = await self.queue.get()

            async with self.job_semaphore:
                await asyncio.to_thread(job.run)
                
            job.status = self.job_status_from_events(job)

            self.queue.task_done() 

            await asyncio.sleep(0.5)