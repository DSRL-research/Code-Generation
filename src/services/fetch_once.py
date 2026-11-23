from collections.abc import Callable

from src.models.rows import Rows


class FetchOnce:
    """Fetches each link once per run, also when several chains under "with" use it.

    The same Rows are given to every chain that reads the link. That is safe because operations never change the
    rows they receive.
    """

    def __init__(self, fetch: Callable[[str], Rows]) -> None:
        self._fetch = fetch
        self._fetched: dict[str, Rows] = {}

    def __call__(self, link: str) -> Rows:
        if link not in self._fetched:
            self._fetched[link] = self._fetch(link)
        return self._fetched[link]
