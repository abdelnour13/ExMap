from dataclasses import dataclass
from typing import Annotated, List
from fastapi import APIRouter, Depends, Header, HTTPException, status
from returns.pipeline import is_successful
from src.shared.platforms import (
    GetBalanceResponse,
    GetModelResponse,
    Platform,
    PlatformFailure,
    PlatformInfo,
    ValidationResponse,
    PlatformsRegistry
)

router = APIRouter(prefix="/llms")

# ---------------------------------------------------------------------------------------------
# Shared checks: the platform exists, and its API key was sent (when it needs one)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    platform: Platform
    api_key: str | None

def resolve_target(
    platform: str,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Target:
    
    found = PlatformsRegistry.create(platform)

    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={ "reason": f"Platform '{platform}' not found. Available: {'|'.join(PlatformsRegistry.available(()))}." },
        )

    if found.requires_api_key and not (x_api_key and x_api_key.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={ "reason": "Header x-api-key is not set." },
        )

    return Target(platform=found, api_key=x_api_key)


# What went wrong on our side of the provider call -> which HTTP status the app sees.
_FAILURE_STATUS = {
    "missing_api_key": status.HTTP_400_BAD_REQUEST,
    "invalid_api_key": status.HTTP_401_UNAUTHORIZED,
    "rate_limited": status.HTTP_429_TOO_MANY_REQUESTS,
    "unreachable": status.HTTP_502_BAD_GATEWAY,
    "provider_error": status.HTTP_502_BAD_GATEWAY,
    "unexpected_response": status.HTTP_502_BAD_GATEWAY,
}


def fail(failure: PlatformFailure) -> HTTPException:

    return HTTPException(
        status_code=_FAILURE_STATUS[failure.kind],
        detail={ "kind": failure.kind, **failure.details },
    )


# ---------------------------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------------------------

@router.get("/available")
def get_apis() -> List[str]:
    return PlatformsRegistry.available()


@router.get("/platforms")
def get_platforms() -> List[PlatformInfo]:
    """The same platforms with what the app needs to know about each (needs a key? has a balance?)."""
    return PlatformInfo.available()  # type: ignore[union-attr]


@router.get("/{platform}/validate_api_key")
async def validate_api_key(
    target: Annotated[Target, Depends(resolve_target)], 
    remote: bool = True
) -> ValidationResponse:
    """`remote=false` only checks the shape of the key and never calls the provider."""
    result = await target.platform.validate_api_key(target.api_key, remote=remote)

    if not is_successful(result):
        raise fail(result.failure())

    return result.unwrap()


@router.get("/{platform}/models")
async def get_models(
    target: Annotated[Target, Depends(resolve_target)]
) -> GetModelResponse:
    
    result = await target.platform.get_models(target.api_key)

    if not is_successful(result):
        raise fail(result.failure())

    return result.unwrap()


@router.get("/{platform}/balance")
async def get_balance(
    target: Annotated[Target, Depends(resolve_target)]
) -> GetBalanceResponse:
    
    result = await target.platform.get_balance(target.api_key)

    if not is_successful(result):
        raise fail(result.failure())

    return result.unwrap()