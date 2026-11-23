"""SourceClient: fetches the link of a source and turns the reply into rows (SourceClient.java)."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from src.clients.records import to_row
from src.clients.replies import error_text, records_in
from src.exceptions import ChainError, SourceError
from src.models.rows import Rows, measure_interval

logger = logging.getLogger(__name__)

_ERROR_BODY_BYTES = 64 * 1024  # how much of a refusal is read, to find its "error"


@dataclass(frozen=True)
class SourceLimits:
    timeout_seconds: float
    max_bytes: int
    max_records: int


class SourceClient:
    """GETs a link and returns its rows, with the interval measured on them.

    | The link ...                                       | Error              |
    |----------------------------------------------------|--------------------|
    | replies 4xx                                        | 400, with its text |
    | replies with a redirect (not followed)             | 400                |
    | replies with something that is not records        | 400                |
    | replies with more records or bytes than the limits | 400                |
    | cannot be reached, replies 5xx, or times out       | 502                |
    """

    def __init__(self, http: httpx.Client, limits: SourceLimits, time_fields: Sequence[str]) -> None:
        self._http = http
        self._limits = limits
        self._time_fields = tuple(time_fields)

    def fetch(self, link: str) -> Rows:
        body = self._get(link)
        try:
            reply = json.loads(body)
        except ValueError:
            raise ChainError(f"{link}: the reply is not JSON") from None
        try:
            records = records_in(reply)
        except ChainError as error:
            error.message = f"{link}: {error.message}"
            raise
        if len(records) > self._limits.max_records:
            raise ChainError(
                f"{link}: {len(records)} records, more than the limit of {self._limits.max_records}; "
                "make the link narrower"
            )
        items = [to_row(record, self._time_fields) for record in records]
        logger.info("fetched %s: %d records", link, len(items))
        return Rows(items, measure_interval(items))

    def _get(self, link: str) -> bytes:
        try:
            with self._http.stream("GET", link, timeout=self._limits.timeout_seconds) as response:
                self._check_status(link, response)
                return self._read(link, response, self._limits.max_bytes)
        except httpx.TimeoutException:
            raise SourceError(f"{link}: no reply after {self._limits.timeout_seconds:g} s") from None
        except (httpx.InvalidURL, httpx.UnsupportedProtocol):
            raise ChainError(f"{link}: not a valid link") from None
        except httpx.HTTPError as error:
            raise SourceError(f"{link}: cannot be reached ({type(error).__name__})") from None

    def _check_status(self, link: str, response: httpx.Response) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        if 300 <= status < 400:
            target = response.headers.get("location", "?")
            raise ChainError(f"{link}: replied with a redirect to {target}; redirects are not followed")
        if 400 <= status < 500:
            body = self._read(link, response, _ERROR_BODY_BYTES, refuse_larger=False)
            raise ChainError(f"{link}: {error_text(body)}")
        raise SourceError(f"{link}: replied with status {status}")

    @staticmethod
    def _read(link: str, response: httpx.Response, max_bytes: int, refuse_larger: bool = True) -> bytes:
        """The body, up to max_bytes. A larger body is an error, or is cut if refuse_larger is False."""
        body = bytearray()
        for chunk in response.iter_bytes():
            body += chunk
            if len(body) > max_bytes:
                if not refuse_larger:
                    return bytes(body[:max_bytes])
                raise ChainError(f"{link}: the reply is larger than {max_bytes} bytes; make the link narrower")
        return bytes(body)
