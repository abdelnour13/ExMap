from fastapi import APIRouter, Path, HTTPException, status
from typing import List
from api.services import graph_learning_service
from api.models import CreateGraphLearningJobBody, GraphLearningJobDetails
from returns.pipeline import is_successful

router = APIRouter(prefix="/experiments")

@router.delete("/{id}")
def delete(id : str = Path()) -> GraphLearningJobDetails:

    response = graph_learning_service.delete(id=id)

    if response.deleted:
        return response.details

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={ "id" : id }
    )

@router.get("")
def all() -> List[GraphLearningJobDetails]:
    return graph_learning_service.get_all()

@router.post("")
def create(body : CreateGraphLearningJobBody) -> GraphLearningJobDetails:

    response = graph_learning_service.create(data=body)

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