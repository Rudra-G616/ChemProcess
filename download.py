"""Download ChEMBL bioactivity assays and their activity measurements.

Example Usage -
python download.py

Methodology -
This script uses the ChEMBL API to find assays relevant to the configured
endpoints, then downloads every activity record associated with those assays.

ChEMBL API documentation - https://www.ebi.ac.uk/chembl/api/data/docs

The data is downloaded at ./data
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import Request, urlopen


BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/"
ASSAY_SEARCH_URL = urljoin(BASE_URL, "assay/search.json")
ACTIVITY_URL = urljoin(BASE_URL, "activity.json")
DATA_DIR = Path("data")
PAGE_SIZE = 1000
ASSAY_ID_BATCH_SIZE = 20
TIMEOUT_SECONDS = 30
MAX_RETRIES = 3

# Add or remove endpoint search terms here. Each key produces one output
# directory; every term is searched and the resulting assays are deduplicated.
ENDPOINTS: dict[str, tuple[str, ...]] = {
    "BBBP": (
        "blood-brain barrier permeability",
        "blood brain barrier permeability",
        "blood-brain barrier penetration",
        "BBB permeability",
    ),
    "Caco2 permeability": (
        "Caco-2 permeability",
        "Caco2 permeability",
        "Caco 2 permeability",
    ),
    "CYP1A2 inhibition": (
        "CYP1A2 inhibition",
        "CYP1A2 inhibitor",
        "cytochrome P450 1A2 inhibition",
    ),
    "CYP2C9 inhibition": (
        "CYP2C9 inhibition",
        "CYP2C9 inhibitor",
        "cytochrome P450 2C9 inhibition",
    ),
    "CYP2C19 inhibition": (
        "CYP2C19 inhibition",
        "CYP2C19 inhibitor",
        "cytochrome P450 2C19 inhibition",
    ),
    "CYP2D6 inhibition": (
        "CYP2D6 inhibition",
        "CYP2D6 inhibitor",
        "cytochrome P450 2D6 inhibition",
    ),
    "CYP3A4 inhibition": (
        "CYP3A4 inhibition",
        "CYP3A4 inhibitor",
        "cytochrome P450 3A4 inhibition",
    ),
    "hERG inhibition": (
        "hERG inhibition",
        "hERG channel inhibition",
        "human ether-a-go-go-related gene inhibition",
        "KCNH2 inhibition",
    ),
    "HLM stability": (
        "human liver microsomal stability",
        "human liver microsome stability",
        "HLM stability",
    ),
    "P-gp substrate": (
        "P-glycoprotein substrate",
        "P glycoprotein substrate",
        "ABCB1 substrate",
        "MDR1 substrate",
    ),
    "RLM stability": (
        "rat liver microsomal stability",
        "rat liver microsome stability",
        "RLM stability",
    ),
}

UrlOpener = Callable[..., Any]


class DownloadError(RuntimeError):
    """Raised when ChEMBL data cannot be downloaded or validated."""


def _request_json(
    url: str,
    *,
    opener: UrlOpener | None = None,
    timeout: int = TIMEOUT_SECONDS,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Request a JSON object, retrying transient HTTP and network failures."""
    open_url = opener or urlopen

    for attempt in range(max_retries + 1):
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ChemProcess/1.0 (+https://github.com/Rudra-G616/ChemProcess)",
            },
        )
        try:
            with open_url(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise DownloadError(f"ChEMBL returned a non-object JSON response for {url}")
            return payload
        except HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == max_retries:
                raise DownloadError(
                    f"ChEMBL request failed with HTTP {exc.code}: {url}"
                ) from exc

            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = float(retry_after) if retry_after is not None else 2**attempt
            except ValueError:
                delay = 2**attempt
            time.sleep(max(0.0, min(delay, 60.0)))
        except (URLError, TimeoutError) as exc:
            if attempt == max_retries:
                raise DownloadError(f"Could not reach ChEMBL: {exc}") from exc
            time.sleep(2**attempt)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DownloadError(f"ChEMBL returned invalid JSON for {url}") from exc

    # The loop always returns or raises. This keeps type checkers satisfied.
    raise AssertionError("unreachable")


def _next_page_url(next_page: object) -> str | None:
    """Resolve and validate a ChEMBL pagination URL."""
    if next_page is None:
        return None
    if not isinstance(next_page, str) or not next_page.strip():
        raise DownloadError("ChEMBL returned an invalid pagination link")

    resolved = urljoin(BASE_URL, next_page)
    expected = urlsplit(BASE_URL)
    actual = urlsplit(resolved)
    if (actual.scheme, actual.netloc) != (expected.scheme, expected.netloc):
        raise DownloadError("ChEMBL returned a pagination link for an unexpected host")
    return resolved


def fetch_assays(
    endpoint: str,
    *,
    opener: UrlOpener | None = None,
    page_size: int = PAGE_SIZE,
) -> list[dict[str, Any]]:
    """Fetch every page of assays returned by a ChEMBL keyword search."""
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError("endpoint search terms must not be empty")
    if page_size < 1:
        raise ValueError("page_size must be positive")

    url: str | None = f"{ASSAY_SEARCH_URL}?{urlencode({'q': endpoint, 'limit': page_size})}"
    assays: list[dict[str, Any]] = []
    visited_urls: set[str] = set()

    while url is not None:
        if url in visited_urls:
            raise DownloadError("ChEMBL returned a circular pagination link")
        visited_urls.add(url)

        payload = _request_json(url, opener=opener)
        page_assays = payload.get("assays")
        if not isinstance(page_assays, list) or not all(
            isinstance(assay, dict) for assay in page_assays
        ):
            raise DownloadError("ChEMBL response is missing a valid 'assays' list")
        assays.extend(page_assays)

        page_meta = payload.get("page_meta")
        if not isinstance(page_meta, dict):
            raise DownloadError("ChEMBL response is missing valid pagination metadata")
        url = _next_page_url(page_meta.get("next"))

    return assays


def fetch_endpoint_assays(
    endpoint: str,
    search_terms: Sequence[str],
    *,
    fetcher: Callable[[str], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Search every synonym for an endpoint and deduplicate the assays."""
    if not search_terms:
        raise ValueError(f"endpoint {endpoint!r} must have at least one search term")

    fetch = fetcher or fetch_assays
    unique_assays: dict[str, dict[str, Any]] = {}

    for search_term in dict.fromkeys(term.strip() for term in search_terms):
        if not search_term:
            raise ValueError(f"endpoint {endpoint!r} contains an empty search term")

        print(f"  Searching for {search_term!r} ...", file=sys.stderr)
        for assay in fetch(search_term):
            assay_id = assay.get("assay_chembl_id")
            if not isinstance(assay_id, str) or not assay_id:
                raise DownloadError(
                    "ChEMBL returned an assay without a valid 'assay_chembl_id'"
                )
            unique_assays.setdefault(assay_id, assay)

    return list(unique_assays.values())


def _batched(values: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
    """Yield fixed-size slices from a sequence."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def iter_activities(
    assay_ids: Sequence[str],
    *,
    opener: UrlOpener | None = None,
    page_size: int = PAGE_SIZE,
    batch_size: int = ASSAY_ID_BATCH_SIZE,
) -> Iterator[dict[str, Any]]:
    """Yield all activities associated with the supplied ChEMBL assay IDs."""
    if page_size < 1:
        raise ValueError("page_size must be positive")

    unique_ids = list(dict.fromkeys(assay_id.strip() for assay_id in assay_ids))
    if any(not assay_id for assay_id in unique_ids):
        raise ValueError("assay IDs must not be empty")

    for assay_id_batch in _batched(unique_ids, batch_size):
        parameters = {
            "assay_chembl_id__in": ",".join(assay_id_batch),
            "limit": page_size,
            "order_by": "activity_id",
        }
        url: str | None = f"{ACTIVITY_URL}?{urlencode(parameters)}"
        visited_urls: set[str] = set()

        while url is not None:
            if url in visited_urls:
                raise DownloadError("ChEMBL returned a circular pagination link")
            visited_urls.add(url)

            payload = _request_json(url, opener=opener)
            page_activities = payload.get("activities")
            if not isinstance(page_activities, list) or not all(
                isinstance(activity, dict) for activity in page_activities
            ):
                raise DownloadError(
                    "ChEMBL response is missing a valid 'activities' list"
                )

            for activity in page_activities:
                returned_assay_id = activity.get("assay_chembl_id")
                if returned_assay_id not in assay_id_batch:
                    raise DownloadError(
                        "ChEMBL returned an activity for an unexpected assay"
                    )
                yield activity

            page_meta = payload.get("page_meta")
            if not isinstance(page_meta, dict):
                raise DownloadError(
                    "ChEMBL activity response is missing valid pagination metadata"
                )
            url = _next_page_url(page_meta.get("next"))


def _safe_name(endpoint: str) -> str:
    """Convert an endpoint name to a safe, readable directory name."""
    normalized = unicodedata.normalize("NFKD", endpoint)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_name).strip("._-")
    return stem or "endpoint"


def save_assays(
    endpoint: str,
    assays: list[dict[str, Any]],
    *,
    search_terms: Sequence[str] | None = None,
    output_dir: Path = DATA_DIR,
) -> Path:
    """Atomically save a search result and return the output path."""
    endpoint_dir = output_dir / _safe_name(endpoint)
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = endpoint_dir / "assays.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    result = {
        "endpoint": endpoint,
        "search_terms": list(search_terms or (endpoint,)),
        "count": len(assays),
        "assays": assays,
    }

    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        temporary.replace(destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise DownloadError(f"Could not write {destination}: {exc}") from exc

    return destination


def save_activities(
    endpoint: str,
    activities: Iterable[dict[str, Any]],
    *,
    output_dir: Path = DATA_DIR,
) -> tuple[Path, int]:
    """Atomically stream activities to JSON Lines and return path and count."""
    endpoint_dir = output_dir / _safe_name(endpoint)
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = endpoint_dir / "activities.jsonl"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    activity_count = 0

    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            for activity in activities:
                json.dump(activity, output_file, ensure_ascii=False, separators=(",", ":"))
                output_file.write("\n")
                activity_count += 1
        temporary.replace(destination)
    except DownloadError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    except (OSError, TypeError, ValueError) as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise DownloadError(f"Could not write {destination}: {exc}") from exc

    return destination, activity_count


def save_manifest(
    endpoint: str,
    search_terms: Sequence[str],
    assay_count: int,
    activity_count: int,
    *,
    output_dir: Path = DATA_DIR,
) -> Path:
    """Save counts and filenames for a completed endpoint download."""
    endpoint_dir = output_dir / _safe_name(endpoint)
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = endpoint_dir / "manifest.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    manifest = {
        "endpoint": endpoint,
        "search_terms": list(search_terms),
        "assay_count": assay_count,
        "activity_count": activity_count,
        "assays_file": "assays.json",
        "activities_file": "activities.jsonl",
    }

    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            json.dump(manifest, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        temporary.replace(destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise DownloadError(f"Could not write {destination}: {exc}") from exc

    return destination


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download assays and activity measurements for the endpoints "
            "configured in download.py."
        )
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the downloader and return a process exit status."""
    build_parser().parse_args(argv)

    try:
        for endpoint, search_terms in ENDPOINTS.items():
            print(f"Searching ChEMBL for endpoint {endpoint!r}:", file=sys.stderr)
            assays = fetch_endpoint_assays(endpoint, search_terms)
            assay_destination = save_assays(
                endpoint,
                assays,
                search_terms=search_terms,
            )
            assay_ids = [assay["assay_chembl_id"] for assay in assays]
            print(
                f"Downloading activities for {len(assay_ids)} assays ...",
                file=sys.stderr,
            )
            activity_destination, activity_count = save_activities(
                endpoint,
                iter_activities(assay_ids),
            )
            save_manifest(
                endpoint,
                search_terms,
                len(assays),
                activity_count,
            )
            print(
                f"Saved {len(assays)} assays to {assay_destination} and "
                f"{activity_count} activities to {activity_destination}"
            )
    except (DownloadError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
