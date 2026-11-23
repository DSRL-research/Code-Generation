import httpx

from src.clients import SourceClient, SourceLimits
from src.config.settings import Settings
from src.policy import ChainChecker, NoPolicy, Policy, PolicyChecker
from src.services.chain_parser import ChainParser
from src.services.chain_service import ChainService
from src.store import ModelStore


def build_chain_service(settings: Settings, http: httpx.Client, models: ModelStore) -> ChainService:
    """Puts the parts of a run together (like a Spring configuration class). To replace a part, change it here."""
    limits = SourceLimits(
        timeout_seconds=settings.fetch_timeout_seconds,
        max_bytes=settings.max_response_bytes,
        max_records=settings.max_records_per_source,
    )
    return ChainService(
        parser=ChainParser(max_expression_length=settings.max_expression_length),
        checker=build_checker(settings),
        client=SourceClient(http, limits, settings.record_time_fields),
        models=models,
    )


def build_checker(settings: Settings) -> ChainChecker:
    """The policy check, or NoPolicy when it is switched off. A policy file that cannot be read stops the service at
    startup."""
    if not settings.policy_enabled:
        return NoPolicy()
    return PolicyChecker(Policy.load(settings.policy_file))
