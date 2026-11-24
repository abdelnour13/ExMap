from fastapi import APIRouter, Depends, Response, Query
from neo4j import Session
from typing import Optional
from api.services import vis_service, graph_service
from api.schemas.graph import GraphSummary, GraphDetail, VisualisationResponse
from api.db.neo4j import get_config, get_session, Neo4jConfig
from api.constants import VIS_DIR, API_URL

router = APIRouter(prefix="/graphs", tags=["graphs"])

def _vis_url(filename: str) -> str:
    return f"{API_URL.rstrip('/')}/public/visualizations/{filename}"


# ── 1. List all graphs ────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=list[GraphSummary],
    summary="List all expertise maps",
)
def list_graphs(
    session: Session = Depends(get_session),
) -> list[GraphSummary]:
    return graph_service.list_graphs(session)


# ── 2. Get graph by ID ────────────────────────────────────────────────────────

@router.get(
    "/{graph_id}",
    response_model=GraphDetail,
    summary="Get full detail for an expertise map",
    responses={404: {"description": "Graph not found"}},
)
def get_graph(
    graph_id: str,
) -> GraphDetail:
    return graph_service.get_graph(graph_id)


# ── 3. Download graph as JSON ─────────────────────────────────────────────────

@router.get(
    "/{graph_id}/download",
    summary="Download the subgraph as JSON via APOC streaming",
    responses={
        200: {"content": {"application/json": {}}},
        404: {"description": "Graph not found"},
        500: {"description": "Export error"},
    },
)
def download_graph(
    graph_id: str,
) -> Response:
    
    json_data = graph_service.download_graph(graph_id)

    return Response(
        content=json_data,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="graph_{graph_id}.json"'
        },
    )


# ── 4. Visualise taxonomy ─────────────────────────────────────────────────────

@router.post(
    "/{graph_id}/vis/taxonomy",
    response_model=VisualisationResponse,
    summary="Generate Domain → SubDomain taxonomy visualisation",
    responses={404: {"description": "Graph not found"}},
)
def vis_taxonomy(
    graph_id: str,
    settings: Neo4jConfig = Depends(get_config),
) -> VisualisationResponse:

    data = graph_service.get_taxonomy_data(graph_id)
    filename = vis_service.build_taxonomy_vis(data)

    return VisualisationResponse(url=_vis_url(settings, filename), filename=filename)


# ── 5. Visualise all expertises ───────────────────────────────────────────────

@router.post(
    "/{graph_id}/vis/expertises",
    response_model=VisualisationResponse,
    summary="Generate Expertise ↔ SubDomain ↔ Domain visualisation",
    responses={404: {"description": "Graph not found"}},
)
def vis_expertises(
    graph_id: str,
    settings: Neo4jConfig = Depends(get_config),
    domains : Optional[list[str]] = Query(default=None),
    subdomains : Optional[list[str]] = Query(default=None),
    labs : Optional[list[str]] = Query(default=None),
) -> VisualisationResponse:

    data = graph_service.get_expertises_data(graph_id, subdomains, domains, labs)
    filename = vis_service.build_expertises_vis(data)

    return VisualisationResponse(url=_vis_url(settings, filename), filename=filename)


# ── 6. Visualise a single expertise ──────────────────────────────────────────

@router.post(
    "/{graph_id}/vis/expertises/{expertise_id}",
    response_model=VisualisationResponse,
    summary="Generate detailed visualisation for a single Expertise",
    responses={
        404: {"description": "Graph or Expertise not found"},
    },
)
def vis_expertise_detail(
    graph_id: str,
    expertise_id: str,
    settings: Neo4jConfig = Depends(get_config),
) -> VisualisationResponse:
    data = graph_service.get_expertise_detail_data(graph_id, expertise_id)
    filename = vis_service.build_expertise_detail_vis(data)
    return VisualisationResponse(url=_vis_url(settings, filename), filename=filename)

# ── 7. Summary statistics visualisation ──────────────────────────────────────

@router.post(
    "/{graph_id}/vis/summary",
    response_model=VisualisationResponse,
    summary="Generate summary statistics bar charts",
    responses={404: {"description": "Graph not found"}},
)
def vis_summary(
    graph_id: str,
    settings: Neo4jConfig = Depends(get_config),
) -> VisualisationResponse:
    data = graph_service.get_summary_data(graph_id)
    filename = vis_service.build_summary_vis(data)
    return VisualisationResponse(url=_vis_url(settings, filename), filename=filename)