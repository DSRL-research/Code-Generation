"""Which links a chain may fetch, and which of them give raw rows (DESIGN.md section 8, AllowedSources.java).

A rule in policy.json: {"match": "http://localhost:8000/microgrids/*/readings*", "raw": true}

The scheme, host and port of a pattern are compared exactly (the port may be `*`); only the path and query are
a glob. So http://localhost:8000* cannot match http://localhost:8000.evil.com.
"""

import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from urllib.parse import urlsplit

from src.exceptions import PolicyFileError, PolicyViolation

_DEFAULT_PORTS = {"http": 80, "https": 443}
_PATTERN = re.compile(r"(?P<scheme>https?)://(?P<host>[^/:*?@\s]+)(?::(?P<port>\d+|\*))?(?P<path>/\S*)")
ANY_PORT = None


@dataclass(frozen=True)
class Link:
    """A link of a chain, split into the parts the rules compare."""

    text: str
    scheme: str
    host: str
    port: int
    path_and_query: str

    @classmethod
    def parse(cls, text: str) -> "Link":
        """Refuses (403) links that could leave a pattern: with user info, `..` or `%` in the path."""
        parts = urlsplit(text)
        try:
            port = parts.port
        except ValueError:
            raise PolicyViolation(f"{text} has a port that cannot be read") from None
        if parts.scheme not in _DEFAULT_PORTS or not parts.hostname:
            raise PolicyViolation(f"{text} is not an http or https link")
        if parts.username is not None or parts.password is not None:
            raise PolicyViolation(f"{text}: links with a user name are not allowed")
        if ".." in parts.path or "%" in parts.path:
            raise PolicyViolation(f"{text}: links with .. or % in the path are not allowed")
        path_and_query = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        return cls(text, parts.scheme, parts.hostname.lower(), port or _DEFAULT_PORTS[parts.scheme], path_and_query)


@dataclass(frozen=True)
class SourceRule:
    """One entry of "sources" in policy.json."""

    match: str
    raw: bool
    scheme: str
    host: str
    port: int | None  # ANY_PORT (None) for `*`
    path: str  # a glob on the path and query

    @classmethod
    def parse(cls, match: str, raw: bool) -> "SourceRule":
        pattern = _PATTERN.fullmatch(match)
        if pattern is None:
            raise PolicyFileError(
                f"source pattern {match!r}: write it as scheme://host[:port]/path, with * only in the port and the "
                "path, for example http://localhost:8000/microgrids/*/readings*"
            )
        scheme, port = pattern["scheme"], pattern["port"]
        if port is None:
            port_number: int | None = _DEFAULT_PORTS[scheme]
        else:
            port_number = ANY_PORT if port == "*" else int(port)
        return cls(match, raw, scheme, pattern["host"].lower(), port_number, pattern["path"])

    def matches(self, link: Link) -> bool:
        return (
            link.scheme == self.scheme
            and link.host == self.host
            and (self.port is ANY_PORT or link.port == self.port)
            and fnmatchcase(link.path_and_query, self.path)
        )
