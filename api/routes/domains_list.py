import os
from fastapi import APIRouter, Body, HTTPException, status
from typing import List
from api.models import GetDomainsList, CreateDomainsList
from api.services import domains_list_service

router = APIRouter(prefix="/domains_list")

@router.get("")
def domains_lists() -> List[GetDomainsList]:
    return domains_list_service.get_all()

@router.post("")
def create_domains_lists(body : CreateDomainsList = Body()) -> GetDomainsList:

    response = domains_list_service.create_domains_lists(data=body)

    if not response.created:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={ 
                "id" : body.name, 
                "reason" : f"domains list with id {body.name} already exists." 
            }
        )

    return response.details