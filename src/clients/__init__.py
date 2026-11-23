"""Sources: fetching a link and turning its reply into rows (DESIGN.md section 4).

- source_client.py  SourceClient: GET a link, with limits; errors as 400 or 502
- replies.py        where the list of records is in a reply
- records.py        one record -> one row, with its time as ts
"""

from src.clients.source_client import SourceClient, SourceLimits

__all__ = ["SourceClient", "SourceLimits"]
