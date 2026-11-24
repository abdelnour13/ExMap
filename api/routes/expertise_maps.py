from fastapi import APIRouter, Path, HTTPException, status
from typing import List
from api.services import expertise_map_service
from api.schemas.models import CreateExpertiseMapBody, ExpertiseMapDetails
from returns.pipeline import is_successful

router = APIRouter(prefix="/maps")

@router.delete("/{id}")
def delete(id : str = Path()) -> ExpertiseMapDetails:

    response = expertise_map_service.delete(id=id)

    if response.deleted:
        return response.details

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={ "id" : id }
    )

@router.get("")
def all() -> List[ExpertiseMapDetails]:
    return expertise_map_service.get_all()

@router.post("")
def create(body : CreateExpertiseMapBody) -> ExpertiseMapDetails:

    response = expertise_map_service.create(data=body)

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