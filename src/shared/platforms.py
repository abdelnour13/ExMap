import asyncio
import re
import httpx
from pydantic import BaseModel, Field
from dataclasses import dataclass
from openai import Client, AsyncClient
from typing import Any, Dict, List, Literal, Type, TypeVar, ClassVar
from openai import APIConnectionError, APIStatusError, AsyncClient, OpenAIError
from functools import cached_property
from src.envirenment import env
from src.shared.prompts import PromptsManager, RequestReturnType
from abc import abstractmethod
from returns.result import Result, Success, Failure
from src.shared.registry import registry

# How long one call to a provider may take before it counts as "unreachable".
REQUEST_TIMEOUT_S = 15.0

# ---------------------------------------------------------------------------------------------
# Failures
#
# A Failure means "we could not find out" (provider unreachable, key missing, provider error...).
# A key the provider REJECTS is not a failure: it is a successful answer, `ValidationResponse(valid=False)`.
# `details` is meant to be spread into the HTTP error body, like the other routers do:
#     detail={"kind": failure.kind, **failure.details}
# It never contains the API key: provider text is scrubbed before it is kept.
# ---------------------------------------------------------------------------------------------

FailureKind = Literal[
    "missing_api_key",      # the platform needs a key and none was given
    "invalid_api_key",      # the provider refused the key (401/403)
    "rate_limited",         # 429
    "unreachable",          # connection error or timeout
    "provider_error",       # any other error status from the provider
    "unexpected_response",  # the provider answered 2xx with a body we do not understand
]

_MESSAGES: Dict[str, str] = {
    "missing_api_key": "This platform needs an API key and none was provided.",
    "invalid_api_key": "The provider rejected this API key.",
    "rate_limited": "The provider is rate limiting requests. Try again in a moment.",
    "unreachable": "The provider could not be reached.",
    "provider_error": "The provider answered with an error.",
    "unexpected_response": "The provider sent a response in an unexpected format.",
}


class PlatformFailure(BaseModel):
    kind: FailureKind
    details: Dict[str, Any] = Field(default_factory=dict)  # always has "platform" and "message"


class ValidateFailure(PlatformFailure):
    ...


class GetModelsFailure(PlatformFailure):
    ...


class GetBalanceFailure(PlatformFailure):
    ...


F = TypeVar("F", bound=PlatformFailure)

_SECRET_LIKE = re.compile(r"(sk-[A-Za-z0-9_\-]{6,}|AIza[0-9A-Za-z_\-]{10,})")


def _scrub(text: str, api_key: str | None) -> str:
    """Provider text that is safe to keep: the key (and anything shaped like one) is masked, length capped."""
    if api_key:
        text = text.replace(api_key, "***")
    return _SECRET_LIKE.sub("***", text)[:300]


class _UnexpectedResponse(Exception):
    """The provider answered successfully with a body that is not what its documentation says."""


def _status_kind(status: int) -> FailureKind:
    if status in (401, 403):
        return "invalid_api_key"
    if status == 429:
        return "rate_limited"
    return "provider_error"


def _failure(cls: Type[F], platform: str, kind: FailureKind, **extra: Any) -> F:
    details: Dict[str, Any] = {"platform": platform, "message": _MESSAGES[kind]}
    details.update({k: v for k, v in extra.items() if v is not None})
    return cls(kind=kind, details=details)


def _failure_from_exception(cls: Type[F], platform: str, exc: Exception, api_key: str | None) -> F:
    """Turns what a provider call can raise into a failure. Programming errors are NOT swallowed."""
    if isinstance(exc, _UnexpectedResponse):
        return _failure(cls, platform, "unexpected_response", provider_message=_scrub(str(exc), api_key))
    if isinstance(exc, APIStatusError):  # openai SDK: 4xx / 5xx
        return _failure(cls, platform, _status_kind(exc.status_code),
                        status_code=exc.status_code, provider_message=_scrub(exc.message, api_key))
    if isinstance(exc, httpx.HTTPStatusError):  # plain httpx calls (balance)
        status = exc.response.status_code
        return _failure(cls, platform, _status_kind(status),
                        status_code=status, provider_message=_scrub(exc.response.text, api_key))
    if isinstance(exc, (APIConnectionError, httpx.TransportError)):  # includes timeouts
        return _failure(cls, platform, "unreachable")
    if isinstance(exc, OpenAIError):
        return _failure(cls, platform, "provider_error", provider_message=_scrub(str(exc), api_key))
    raise exc


# ---------------------------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------------------------

class ValidationResponse(BaseModel):
    valid: bool
    # How much the answer can be trusted:
    #   "assumed"  : not checked, assumed valid (the platform does not need a key)
    #   "format"   : only the shape of the key was checked, with a regex
    #   "verified" : a request was made to the provider and its answer decided
    validation_level: Literal["assumed", "format", "verified"]
    reason: str | None = None  # why it is invalid, when it is


class GetModelResponse(BaseModel):
    models: List[str] = Field(default_factory=list)
    available: bool  # False if for the specific platform we can't get the available models: the list is empty


class GetBalanceResponse(BaseModel):
    amount: float | None = None
    currency: str | None = None  # "USD", "CNY"...
    available: bool  # False if the platform has no way to report a balance: amount is None


class PlatformInfo(BaseModel):
    name: str
    base_url: str
    requires_api_key: bool
    supports_balance: bool = False

    @staticmethod
    def available() -> List['PlatformInfo']:
        return list(__PLATFORMS__.values())

__PLATFORMS__: Dict[str, PlatformInfo] = {
    "deepseek": PlatformInfo(name="deepseek", base_url="https://api.deepseek.com", requires_api_key=True, supports_balance=True),
    "gemini": PlatformInfo(name="gemini", base_url="https://generativelanguage.googleapis.com/v1beta/openai/", requires_api_key=True),
    "openai": PlatformInfo(name="openai", base_url="https://api.openai.com/v1", requires_api_key=True),
}

PlatformType = Literal['deepseek', 'gemini', 'openai']

# ---------------------------------------------------------------------------------------------
# Platforms
# ---------------------------------------------------------------------------------------------

class Platform(BaseModel):
    name : PlatformType

    # Shape of a valid key, used for the cheap "format" check. None: the shape is not checked.
    KEY_PATTERN: ClassVar[re.Pattern[str] | None] = None

    @property
    def info(self) -> PlatformInfo:
        return __PLATFORMS__[self.name]

    @property
    def base_url(self) -> str:
        return self.info.base_url

    @property
    def requires_api_key(self) -> bool:
        return self.info.requires_api_key

    @property
    def supports_balance(self) -> bool:
        return self.info.supports_balance

    def check_key_format(self, api_key: str) -> bool:
        return self.KEY_PATTERN is None or self.KEY_PATTERN.fullmatch(api_key) is not None

    @abstractmethod
    async def validate_api_key(self, api_key: str | None, *, remote: bool = True) -> Result[ValidationResponse, ValidateFailure]:
        """Is this key good? `remote=False` only checks its shape (no request is made)."""
        ...

    @abstractmethod
    async def get_models(self, api_key: str | None) -> Result[GetModelResponse, GetModelsFailure]:
        ...

    @abstractmethod
    async def get_balance(self, api_key: str | None) -> Result[GetBalanceResponse, GetBalanceFailure]:
        ...


class OpenAICompatiblePlatform(Platform):
    """A platform that speaks the OpenAI API (`GET {base_url}/models`). Subclasses adjust the details."""

    def _client(self, api_key: str | None) -> AsyncClient:
        # max_retries=0: a failing provider must answer the user quickly, not after hidden retries.
        return AsyncClient(api_key=api_key or "not-needed", base_url=self.base_url, timeout=REQUEST_TIMEOUT_S, max_retries=0)

    def _is_key_rejection(self, exc: APIStatusError) -> bool:
        return exc.status_code in (401, 403)

    def _normalize_model_id(self, model_id: str) -> str:
        return model_id

    def _keep_model(self, model_id: str) -> bool:
        """Only models that can chat are useful here (not embeddings, speech, images...)."""
        return True

    async def _list_model_ids(self, api_key: str | None) -> List[str]:
        async with self._client(api_key) as client:
            ids = [model.id async for model in client.models.list()]
        names = {self._normalize_model_id(i) for i in ids}
        return sorted(name for name in names if self._keep_model(name))

    async def validate_api_key(self, api_key: str | None, *, remote: bool = True) -> Result[ValidationResponse, ValidateFailure]:
        if not self.requires_api_key:
            return Success(ValidationResponse(valid=True, validation_level="assumed"))

        key = (api_key or "").strip()
        if not key:
            return Failure(_failure(ValidateFailure, self.name, "missing_api_key"))

        if not self.check_key_format(key):
            return Success(ValidationResponse(valid=False, validation_level="format",
                                              reason="This does not look like a key of this platform."))
        if not remote:
            return Success(ValidationResponse(valid=True, validation_level="format"))

        # Listing the models is the cheapest authenticated call: it succeeds only with a good key.
        try:
            async with self._client(key) as client:
                await client.models.list()
        except APIStatusError as exc:
            if self._is_key_rejection(exc):
                return Success(ValidationResponse(valid=False, validation_level="verified", reason=_MESSAGES["invalid_api_key"]))
            return Failure(_failure_from_exception(ValidateFailure, self.name, exc, key))
        except (OpenAIError, httpx.HTTPError) as exc:
            return Failure(_failure_from_exception(ValidateFailure, self.name, exc, key))
        return Success(ValidationResponse(valid=True, validation_level="verified"))

    async def get_models(self, api_key: str | None) -> Result[GetModelResponse, GetModelsFailure]:
        key = (api_key or "").strip()
        if self.requires_api_key and not key:
            return Failure(_failure(GetModelsFailure, self.name, "missing_api_key"))
        try:
            models = await self._list_model_ids(key)
        except (OpenAIError, httpx.HTTPError) as exc:
            return Failure(_failure_from_exception(GetModelsFailure, self.name, exc, key))
        return Success(GetModelResponse(models=models, available=True))

    async def get_balance(self, api_key: str | None) -> Result[GetBalanceResponse, GetBalanceFailure]:
        # Most platforms have no balance endpoint for a normal API key.
        return Success(GetBalanceResponse(amount=None, currency=None, available=False))


PlatformsRegistry = registry(Platform, cached=True)


@PlatformsRegistry.register("deepseek")
class DeepSeekPlatform(OpenAICompatiblePlatform):
    KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9]{20,}")

    def __init__(self) -> None:
        super().__init__(name='deepseek')

    async def get_balance(self, api_key: str | None) -> Result[GetBalanceResponse, GetBalanceFailure]:
        """GET {base_url}/user/balance -> {"is_available": bool, "balance_infos": [{"currency", "total_balance", ...}]}"""
        key = (api_key or "").strip()
        if not key:
            return Failure(_failure(GetBalanceFailure, self.name, "missing_api_key"))
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as http:
                response = await http.get(
                    f"{self.base_url}/user/balance",
                    headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                )
                response.raise_for_status()
                try:
                    body = response.json()
                except ValueError as exc:
                    raise _UnexpectedResponse("the balance is not JSON") from exc
            return Success(self._parse_balance(body))
        except (_UnexpectedResponse, httpx.HTTPError) as exc:
            return Failure(_failure_from_exception(GetBalanceFailure, self.name, exc, key))

    @staticmethod
    def _parse_balance(body: Any) -> GetBalanceResponse:
        infos = body.get("balance_infos") if isinstance(body, dict) else None
        if not isinstance(infos, list) or not infos:
            raise _UnexpectedResponse("no balance_infos in the answer")
        entries: List[tuple[str, float]] = []
        for info in infos:
            try:
                entries.append((str(info["currency"]), float(info["total_balance"])))  # the amount is sent as a string
            except (KeyError, TypeError, ValueError) as exc:
                raise _UnexpectedResponse("a balance entry is malformed") from exc
        # An account may hold several currencies; the app shows one: USD if there is one, else the first.
        currency, amount = next((e for e in entries if e[0] == "USD"), entries[0])
        return GetBalanceResponse(amount=amount, currency=currency, available=True)


@PlatformsRegistry.register("openai")
class OpenAIPlatform(OpenAICompatiblePlatform):
    KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")  # also matches "sk-proj-..."

    _EXCLUDED: ClassVar[tuple[str, ...]] = ("audio", "realtime", "transcribe", "tts", "image", "embedding", "moderation", "instruct", "whisper", "dall-e", "search")

    def __init__(self) -> None:
        super().__init__(name='openai')

    def _keep_model(self, model_id: str) -> bool:
        is_chat_family = re.match(r"(gpt-|chatgpt-|o\d)", model_id) is not None
        return is_chat_family and not any(word in model_id for word in self._EXCLUDED)


@PlatformsRegistry.register("gemini")
class GeminiPlatform(OpenAICompatiblePlatform):
    KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{30,}")

    _EXCLUDED: ClassVar[tuple[str, ...]] = ("embedding", "aqa", "tts", "image", "imagen", "veo", "live")

    def __init__(self) -> None:
        super().__init__(name='gemini')

    def _is_key_rejection(self, exc: APIStatusError) -> bool:
        # Google answers a wrong key with 400 "API key not valid" rather than 401.
        return super()._is_key_rejection(exc) or (exc.status_code == 400 and "api key" in exc.message.lower())

    def _normalize_model_id(self, model_id: str) -> str:
        return model_id.removeprefix("models/")  # listed as "models/gemini-...", used as "gemini-..."

    def _keep_model(self, model_id: str) -> bool:
        return model_id.startswith("gemini") and not any(word in model_id for word in self._EXCLUDED)


# ---------------------------------------------------------------------------------------------
# LLM Class
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LLM:
    name : str
    api_key : str = None # Local models don't need it

    @staticmethod
    def available_platforms() -> List[PlatformInfo]:
        return list(__PLATFORMS__.values())

    def __post_init__(self) -> None:

        if self.api_key is None:

            if not self.platform.requires_api_key:
                return object.__setattr__(self, 'api_key', "no-need")

            api_key = env.get_api_key(self.platform.name)

            if api_key is None:
                raise Exception(f"Api key for platform {self.platform.name} is neither provided nor it is registered in `env.json`.")

            object.__setattr__(self, 'api_key', api_key)


    @property
    def platform(self) -> Platform:
        name = self.name.split('-')[0]
        return PlatformsRegistry.create(name)

    @property
    def url(self) -> str:
        return self.platform.base_url

    @cached_property
    def client(self) -> Client:

        return Client(
            api_key=self.api_key,
            base_url=self.url
        )

    @cached_property
    def async_client(self) -> Client:

        return AsyncClient(
            api_key=self.api_key,
            base_url=self.url
        )

    async def make_async_request(self, 
        prompt_name: str,
        input: Dict[str, Any],
        lang: str = 'en',
        validate_input: bool = True,
        validate_output: bool = True                 
    ) -> RequestReturnType[Dict]:

        result = await PromptsManager.make_request(
            prompt_name=prompt_name,
            client=self.async_client,
            model_name=self.name,
            input=input,
            lang=lang,
            validate_input=validate_input,
            validate_output=validate_output
        )

        return result

    async def _gather_make_async_request(self, 
        prompt_name: str,
        inputs: List[Dict[str, Any]],
        lang: str = 'en',
        validate_input: bool = True,
        validate_output: bool = True                 
    ) -> List[RequestReturnType[Dict]]:

        result = await asyncio.gather(*[
            PromptsManager.make_request(
                prompt_name=prompt_name,
                client=self.async_client,
                model_name=self.name,
                input=input,
                lang=lang,
                validate_input=validate_input,
                validate_output=validate_output
            )
            for input in inputs
        ])

        return result

    def batch_make_async_request(self, 
        prompt_name: str,
        inputs: List[Dict[str, Any]],
        lang: str = 'en',
        validate_input: bool = True,
        validate_output: bool = True                 
    ) -> List[RequestReturnType[Dict]]:
        
        return asyncio.run(self._gather_make_async_request(
            prompt_name=prompt_name,
            inputs=inputs,
            lang=lang,
            validate_output=validate_output,
            validate_input=validate_input
        ))