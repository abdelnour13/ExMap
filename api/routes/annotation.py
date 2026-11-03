import os
import shutil
from fastapi import APIRouter, Path, HTTPException, status
from typing import List
from api.services import annotation_serivce
from api.models import AnnotationDetails, CreateAnnotationBody
from returns.pipeline import is_successful

router = APIRouter(prefix="/annotations")

@router.delete("/{id}")
def delete_annotation(id : str = Path()) -> AnnotationDetails:

    response = annotation_serivce.delete(id=id)

    if response.deleted:
        return response.details

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={ "id" : id }
    )

@router.get("")
def  annotations() -> List[AnnotationDetails]:
    return annotation_serivce.get_all()

@router.post("")
def create_annotations(body : CreateAnnotationBody) -> AnnotationDetails:

    response = annotation_serivce.create(data=body)

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