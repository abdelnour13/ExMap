from fastapi import APIRouter, HTTPException, status, Body, Path
from typing import List
from api.schemas.models import CreateCorpusBody, CorpusDetails
from api.services import corpuses_service

router = APIRouter(prefix="/corpuses")

@router.get("/apis")
def get_apis() -> List[str]:
    return ["hal", "pubmedia"]

@router.post("")
def create_corpus(body : CreateCorpusBody = Body()) -> CorpusDetails:

    create_response = corpuses_service.create_corpus(data=body)

    if not create_response.created:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={ "id" : body.id, "reason" : f"corpus with id {body.id} already exists." }
        )

    return create_response.details
    
@router.delete("/{corpus_id}")
def delete_corupus(corpus_id : str = Path()) -> CorpusDetails:

    delete_response = corpuses_service.delete_corpus(id=corpus_id)

    if delete_response.deleted:
        return delete_response.details

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={ "id" : corpus_id }
    )

@router.get("")
def corpuses() -> List[CorpusDetails]:
    return corpuses_service.get_all()