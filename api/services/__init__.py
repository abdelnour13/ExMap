from .corpuses import CorpusesService
from .domains_list import DomainsListService
from .scheduler import Scheduler
from .annotations import AnnotationService
from .knowledge_graph_construction import KnowledgeGraphConstructionService

scheduler_service = Scheduler(n=1)
corpuses_service = CorpusesService(scheduler=scheduler_service)
domains_list_service = DomainsListService()

annotation_serivce = AnnotationService(
    scheduler=scheduler_service,
    courpses_service=corpuses_service,
    domains_list_service=domains_list_service
)

kg_construction_service = KnowledgeGraphConstructionService(
    scheduler=scheduler_service,
    annotation_service=annotation_serivce
)