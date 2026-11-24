from fastapi import APIRouter, Path, HTTPException, status
from typing import List
from api.services import kg_construction_service
from api.schemas.models import KnowledgeGraphDetails, CreateKnowledgeGraphBody
from returns.pipeline import is_successful

router = APIRouter(prefix="/kgs")

@router.delete("/{id}")
def delete_kg(id : str = Path()) -> KnowledgeGraphDetails:

    response = kg_construction_service.delete(id=id)

    if response.deleted:
        return response.details

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={ "id" : id }
    )

@router.get("")
def kgs() -> List[KnowledgeGraphDetails]:
    return kg_construction_service.get_all()

@router.post("")
def create_kg(body : CreateKnowledgeGraphBody) -> KnowledgeGraphDetails:

    response = kg_construction_service.create(data=body)

    if not is_successful(response):

        failure = response.failure()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={ 
                "id" : body.id,
                "kind" : failure.kind,
                **failure.details
            }
        )

    return response.unwrap()