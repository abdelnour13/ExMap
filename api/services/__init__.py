from .corpuses import CorpusesService
from .domains_list import DomainsListService
from .scheduler import Scheduler
from .annotations import AnnotationService
from .knowledge_graph_construction import KnowledgeGraphConstructionService
from .feature_extraction import FeaturesExtractionService
from .graph_learning import GraphLearningService
from .expertise_maps import ExpertiseMapService
from .graph import GraphService
from .vis import VisualizationService
from api.db.neo4j import get_driver

scheduler_service = Scheduler(n=1)
graph_service = GraphService(driver=get_driver())
vis_service = VisualizationService()
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

feature_extraction_service = FeaturesExtractionService(
    scheduler=scheduler_service,
    kg_service=kg_construction_service
)

graph_learning_service = GraphLearningService(
    scheduler=scheduler_service,
    features_service=feature_extraction_service
)

expertise_map_service = ExpertiseMapService(
    scheduler=scheduler_service,
    gl_service=graph_learning_service,
    graph_service=graph_service
)