"""Simple Lens.org Patent API client for assignee-based counts.

Notes:
- Requires `LENS_API_TOKEN` in environment (config.LENS_API_TOKEN)
- Endpoint default: https://api.lens.org/patent/search

We query per organization with a phrase match on assignee/owner fields and
request size=0 so we only get a count. Lens returns a `total` field in `data`
or a top-level `total` depending on tier/version. This client normalizes into
an integer count.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
import requests
import time
import os
import re
import unicodedata

logger = logging.getLogger(__name__)


class LensPatentClient:
    def __init__(self, api_url: str, token: str):
        self.api_url = api_url.rstrip('/')
        self.token = token
        self.session = requests.Session()
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}',
        }
        # Tunables
        self._max_attempts = int(os.getenv('LENS_MAX_ATTEMPTS', '5'))
        self._base_backoff = float(os.getenv('LENS_BASE_BACKOFF_SEC', '1.0'))

    def _build_query_payload(self, variants: list[str]) -> dict:
        # Prefer query_string with ORed exact phrases across common fields
        field_pairs = ["owners.name", "assignees.name", "applicants.name"]
        parts = []
        for v in variants:
            phrase = v.replace('"', '')
            for fld in field_pairs:
                parts.append(f'{fld}:"{phrase}"')
        qs = ' OR '.join(parts)
        return {
            "query": {
                "query_string": {
                    "query": qs
                }
            },
            "size": 0,
        }

    @staticmethod
    def normalize_org_name(name: str) -> str:
        s = name or ''
        s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
        s = s.replace('&', ' and ')
        s = re.sub(r'[^\w\s-]', ' ', s)
        s = ' '.join(s.split())
        return s.strip()

    @staticmethod
    def generate_variations(name: str) -> list[str]:
        base = LensPatentClient.normalize_org_name(name)
        # Strip common corporate suffixes
        suffixes = [
            'inc', 'llc', 'ltd', 'corp', 'co', 'company', 'corporation', 'plc', 'ag', 'nv', 'bv',
            'oy', 'ab', 'sarl', 'gmbh', 'sa', 'sas', 'spa', 'kk', 'kabushiki kaisha', 'pte', 'limited'
        ]
        tokens = base.split()
        def strip_suffix(tok_list):
            while tok_list and tok_list[-1].lower().strip('.') in suffixes:
                tok_list = tok_list[:-1]
            return tok_list
        stripped = ' '.join(strip_suffix(tokens))
        variants = {name.strip() for name in [name, base, stripped] if name}
        # & vs and handled in normalize; generate alt by substituting and -> &
        if ' and ' in base.lower():
            variants.add(base.lower().replace(' and ', ' & ').title())
        # Short two-word prefix for partial legal names
        words = stripped.split()
        if len(words) >= 2:
            variants.add(' '.join(words[:2]))
        # Known expansions
        expansions = {
            'intl': 'international',
            'tech': 'technologies',
            'sys': 'systems',
        }
        for abbr, full in expansions.items():
            if f' {abbr} ' in f' {stripped.lower()} ':
                variants.add(stripped.lower().replace(f' {abbr} ', f' {full} ').title())
        # Known synonym expansions for local/regional frequent companies
        synonyms = {
            'h r block': ['H&R Block', 'HRB Tax Group', 'H and R Block'],
            'black veatch': ['Black & Veatch', 'Black & Veatch Corporation'],
            'burns mcdonnell': ['Burns & McDonnell', 'Burns & McDonnell Engineering'],
            'childrens mercy hospital': ["Children's Mercy Hospital", 'Childrens Mercy Hospital'],
            'st lukes health system': ["Saint Luke's Health System", "St. Luke's Health System"],
            'garmin': ['Garmin Ltd', 'Garmin International'],
            'cerner': ['Cerner', 'Cerner Corporation', 'Oracle Health'],
            'yrc worldwide': ['YRC Worldwide', 'Yellow Corporation'],
            'dst systems': ['DST Systems', 'SS&C Technologies'],
            'hallmark cards': ['Hallmark', 'Hallmark Cards Inc', 'Hallmark Cards, Inc.'],
        }
        key = base.lower()
        for k, syns in synonyms.items():
            if k in key:
                variants.update(syns)
        return [v for v in variants if v]

    def search_assignee_count(self, org_name: str, timeout: int = 20) -> int:
        try:
            variants = self.generate_variations(org_name)
            # Try batched OR query
            payload = self._build_query_payload(variants)
            r = self._post_with_backoff(payload, timeout=timeout)
            if r is not None and r.status_code == 200:
                data = r.json() if r.content else {}
                if isinstance(data, dict):
                    if 'data' in data and isinstance(data['data'], dict) and 'total' in data['data']:
                        return int(data['data']['total'] or 0)
                    if 'total' in data:
                        return int(data.get('total') or 0)
            elif r is not None:
                logger.warning(f"Lens API error for {org_name}: HTTP {r.status_code}")
            # Fallback: try each variant individually if batch returned 0/unknown
            for v in variants:
                payload_v = self._build_query_payload([v])
                rv = self._post_with_backoff(payload_v, timeout=timeout)
                if rv is not None and rv.status_code == 200:
                    jd = rv.json() if rv.content else {}
                    if 'data' in jd and isinstance(jd['data'], dict) and 'total' in jd['data']:
                        if int(jd['data']['total'] or 0) > 0:
                            return int(jd['data']['total'])
                    if 'total' in jd and int(jd.get('total') or 0) > 0:
                        return int(jd.get('total') or 0)
            return 0
        except Exception as e:
            logger.debug(f"Lens request failed for {org_name}: {e}")
            return 0

    def search_batch_counts(self, org_names: List[str], per_request_sleep: float = None) -> Dict[str, int]:
        import time
        results: Dict[str, int] = {}
        # Deduplicate by normalized key to avoid repeated calls for obvious duplicates
        cache: dict[str, int] = {}
        if per_request_sleep is None:
            # Allow override via env; default to conservative pacing to avoid 429
            per_request_sleep = float(os.getenv('LENS_PER_REQUEST_SLEEP_SEC', '0.8'))
        for name in org_names:
            key = self.normalize_org_name(name).lower()
            if key in cache:
                results[name] = cache[key]
                continue
            count = self.search_assignee_count(name)
            results[name] = count
            cache[key] = count
            time.sleep(per_request_sleep)
        return results

    def _post_with_backoff(self, payload: dict, timeout: int = 20) -> Optional[requests.Response]:
        """POST with exponential backoff and 429 Retry-After handling."""
        backoff = self._base_backoff
        attempts = self._max_attempts
        last_resp: Optional[requests.Response] = None
        for _ in range(attempts):
            try:
                r = self.session.post(self.api_url, headers=self.headers, json=payload, timeout=timeout)
                last_resp = r
                if r.status_code == 429:
                    retry_after = r.headers.get('Retry-After')
                    try:
                        sleep_s = float(retry_after)
                    except Exception:
                        sleep_s = backoff
                    logger.warning("Lens API rate limited (429). Backing off for %.2fs", sleep_s)
                    time.sleep(max(sleep_s, 0.5))
                    backoff = min(backoff * 1.8, 10.0)
                    continue
                if r.status_code >= 500:
                    time.sleep(backoff)
                    backoff = min(backoff * 1.8, 10.0)
                    continue
                return r
            except requests.RequestException:
                time.sleep(backoff)
                backoff = min(backoff * 1.8, 10.0)
        return last_resp
