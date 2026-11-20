from fastapi import APIRouter, Path, HTTPException, status
from typing import List
from api.services import feature_extraction_service
from api.models import CreateFeaturesExtractionJobBody, FeatureExtractionJobDetails
from returns.pipeline import is_successful

router = APIRouter(prefix="/features")

@router.delete("/{id}")
def delete(id : str = Path()) -> FeatureExtractionJobDetails:

    response = feature_extraction_service.delete(id=id)

    if response.deleted:
        return response.details

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={ "id" : id }
    )

@router.get("")
def all() -> List[FeatureExtractionJobDetails]:
    return feature_extraction_service.get_all()

@router.post("")
def create(body : CreateFeaturesExtractionJobBody) -> FeatureExtractionJobDetails:

    response = feature_extraction_service.create(data=body)

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