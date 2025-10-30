"""Batch patent search optimization for KC Cluster Prediction Tool

This module previously used the PatentsView API (search.patentsview.org),
which now frequently returns 410/500 responses. To avoid noisy failures
and wasted time, we disable external patent fetches by default when the
configured endpoint appears to be PatentsView or when SKIP_PATENT_SEARCH
is enabled. In that case, we immediately return zero counts with a clear log.

To explicitly force PatentsView on (not recommended), set env
ENABLE_PATENTSVIEW=true.
"""
import json
import logging
import time
from typing import Dict, List, Set, Tuple, Optional
import requests
from dataclasses import dataclass
from utils.circuit_breaker import circuit_breaker
from .lens_client import LensPatentClient
import re
from collections import defaultdict
import os

logger = logging.getLogger(__name__)

@dataclass
class PatentData:
    """Structure for patent information"""
    patent_id: str
    title: str
    assignee_org: str
    assignee_city: str
    assignee_state: str

class BatchPatentSearcher:
    """Optimized patent search using batch location queries"""
    
    def __init__(self, config):
        self.config = config
        self.api_url = f"{config.USPTO_API_URL}patent"
        self.headers = {
            'Accept': 'application/json',
            'X-Api-Key': config.USPTO_API_KEY
        }
        self._progress_cb = None
        # KC area cities for location-based search (moved from _emit_progress to init)
        self.kc_cities: List[str] = [
            "Kansas City", "Overland Park", "Olathe", "Independence",
            "Lee's Summit", "Shawnee", "Blue Springs", "Lenexa",
            "Leavenworth", "Liberty", "Raytown", "Gladstone",
            "Prairie Village", "Gardner", "Grandview", "Leawood",
            "Mission", "Raymore", "Belton", "Grain Valley"
        ]
        # Decide if external patent API should be disabled
        url_lower = str(self.api_url).lower()
        patentsview_like = ('patentsview' in url_lower)
        enable_patentsview = os.getenv('ENABLE_PATENTSVIEW', 'false').lower() == 'true'
        skip_patent_search = bool(getattr(self.config, 'SKIP_PATENT_SEARCH', False))
        # Lens client (preferred)
        lens_token = getattr(self.config, 'LENS_API_TOKEN', None) or os.getenv('LENS_API_TOKEN')
        lens_url = getattr(self.config, 'LENS_API_URL', None) or os.getenv('LENS_API_URL', 'https://api.lens.org/patent/search')
        self.lens: Optional[LensPatentClient] = None
        if lens_token:
            try:
                self.lens = LensPatentClient(api_url=lens_url, token=lens_token)
                logger.info("Lens API enabled for patent counts")
            except Exception as e:
                logger.warning(f"Failed to initialize Lens client: {e}")
                self.lens = None
        # Disable PatentsView if present and not explicitly enabled
        self._patents_api_disabled = skip_patent_search or (patentsview_like and not enable_patentsview)

        logger.info(
            f"BatchPatentSearcher module path: {__file__}, uspto_url={self.api_url}, "
            f"skip_patent_search={skip_patent_search}, patents_api_disabled={self._patents_api_disabled}"
        )
    
    def set_progress_callback(self, cb):
        """Provide a progress callback receiving (percent:int, message:str)."""
        self._progress_cb = cb
    
    def _emit_progress(self, pct: int, message: str):
        try:
            if self._progress_cb:
                self._progress_cb(int(max(0, min(100, pct))), message)
        except Exception:
            pass
        
    def batch_search_patents(self, business_names: List[str]) -> Dict[str, int]:
        """
        Perform batch patent search for all businesses
        
        Args:
            business_names: List of business names to search
            
        Returns:
            Dictionary mapping business name to patent count
        """
        logger.info(f"Starting batch patent search for {len(business_names)} businesses")
        # Prefer Lens if available
        if self.lens:
            return self.lens.search_batch_counts(business_names)
        # Short-circuit if PatentsView is disabled/unavailable
        if self._patents_api_disabled or not self.headers.get('X-Api-Key'):
            if self._patents_api_disabled:
                logger.warning("Patent API disabled or unsupported (PatentsView detected); returning zero counts.")
            else:
                logger.warning("No USPTO API key found; returning zero patent counts.")
            return {name: 0 for name in business_names}
        start_time = time.time()
        
        # Step 1: Fetch all KC-area patents
        all_patents = self._fetch_all_kc_area_patents()
        fetch_time = time.time() - start_time
        logger.info(f"Fetched {len(all_patents)} KC-area patents in {fetch_time:.2f} seconds")
        
        # Step 2: Build searchable indexes
        patent_index = self._build_patent_index(all_patents)
        
        # Step 3: Match businesses to patents
        patent_counts = self._match_businesses_to_patents(business_names, patent_index)
        
        total_time = time.time() - start_time
        matched_count = sum(1 for count in patent_counts.values() if count > 0)
        logger.info(f"Batch search complete in {total_time:.2f} seconds")
        logger.info(f"Found patents for {matched_count}/{len(business_names)} businesses")
        
        return patent_counts
    
    def _fetch_all_kc_area_patents(self) -> List[PatentData]:
        """Fetch all patents from KC metropolitan area via POST pagination + retries.
        Falls back to company-only if location query keeps failing.
        """
        if self._patents_api_disabled:
            return []
        all_patents: List[PatentData] = []
        seen_patent_ids: Set[str] = set()

        location_query = {
            "_and": [
                {"_or": [{"assignees.assignee_city": city} for city in self.kc_cities]},
                {"_or": [{"assignees.assignee_state": "MO"}, {"assignees.assignee_state": "KS"}]}
            ]
        }

        # POST pagination params
        per_page = 200
        max_pages = 50  # cap ~10k records
        url = self.api_url if self.api_url.endswith('/') else self.api_url + '/'
        self._emit_progress(15, f"Starting KC-area patent fetch (per_page={per_page}, max_pages={max_pages})")

        @circuit_breaker(failure_threshold=3, recovery_timeout=60, name="patentsview_api")
        def _cb_post(url: str, payload: Dict):
            return requests.post(url, json=payload, headers={**self.headers, 'Content-Type': 'application/json'}, timeout=30)

        def post_with_retry(payload: Dict, retries: int = 3, backoff: float = 1.0):
            for attempt in range(retries):
                try:
                    resp = _cb_post(url, payload)
                    if resp.status_code == 200:
                        return resp
                    logger.warning(f"PatentsView POST failed (status {resp.status_code}) on page {payload.get('o',{}).get('page')}.")
                except Exception as e:
                    logger.warning(f"PatentsView POST exception on page {payload.get('o',{}).get('page')}: {e}")
                time.sleep(backoff)
                backoff *= 2
            return None

        pages_ok = 0
        for page in range(1, max_pages + 1):
            self._emit_progress(min(90, 15 + int((page / max_pages) * 70)), f"Fetching KC patents page {page}/{max_pages}")
            payload = {
                "q": location_query,
                "f": ["patent_id", "patent_title", "assignees"],
                "o": {"per_page": per_page, "page": page, "exclude_withdrawn": True}
            }
            resp = post_with_retry(payload)
            if not resp:
                if page == 1:
                    logger.error("KC location query failed repeatedly; degrading to company-only patents.")
                    break
                else:
                    logger.warning("Stopping KC pagination due to repeated failures.")
                    break

            data = resp.json() if resp.content else {}
            patents = data.get('patents', [])
            if not patents:
                logger.info("KC pagination returned no results; stopping.")
                break

            for patent in patents:
                pid = patent.get('patent_id')
                if not pid or pid in seen_patent_ids:
                    continue
                # Verify any KS/MO assignee present
                for assignee in (patent.get('assignees') or []):
                    if assignee.get('assignee_state') in ['KS', 'MO'] and assignee.get('assignee_organization'):
                        seen_patent_ids.add(pid)
                        all_patents.append(PatentData(
                            patent_id=pid,
                            title=patent.get('patent_title', ''),
                            assignee_org=assignee.get('assignee_organization',''),
                            assignee_city=assignee.get('assignee_city',''),
                            assignee_state=assignee.get('assignee_state','')
                        ))
                        break

            pages_ok += 1
            # gentle rate limit
            time.sleep(0.6)

        self._emit_progress(95, f"KC-area patent fetch complete: {len(seen_patent_ids):,} unique")
        # Also fetch major KC companies' patents nationwide
        major_companies = [
            "Cerner", "Garmin", "Black Veatch", "Burns McDonnell",
            "Honeywell", "Hallmark", "YRC Worldwide", "DST Systems",
            "H&R Block", "American Century", "Sprint", "T-Mobile"
        ]
        
        logger.info("Fetching patents for major KC companies")
        for company in major_companies:
            company_patents = self._fetch_company_patents_nationwide(company)
            all_patents.extend(company_patents)
            time.sleep(1.5)  # Rate limiting
        
        # All patent tasks complete
        self._emit_progress(100, "Patent analysis complete")
        return all_patents

    def batch_search_patents_for_orgs(self, org_names: List[str]) -> Dict[str, int]:
        """Fast path: fetch patents only for the provided organization names (nationwide)."""
        # Prefer Lens if available
        if self.lens:
            return self.lens.search_batch_counts(org_names)
        # Short-circuit if PatentsView is disabled/unavailable
        if self._patents_api_disabled or not self.headers.get('X-Api-Key'):
            if self._patents_api_disabled:
                logger.warning("Patent API disabled or unsupported (PatentsView detected); returning zero counts for orgs.")
            else:
                logger.warning("No USPTO API key found; returning zero patent counts for orgs.")
            return {name: 0 for name in org_names}

        counts: Dict[str, int] = {}
        total = max(1, len(org_names))
        for i, name in enumerate(org_names, 1):
            self._emit_progress(int(15 + (i / total) * 80), f"Fetching patents for {name} [{i}/{total}]")
            try:
                patents = self._fetch_company_patents_nationwide(name)
                counts[name] = len(patents)
            except Exception:
                counts[name] = 0
        self._emit_progress(100, "Organization-specific patent fetch complete")
        return counts
    
    def _fetch_company_patents_nationwide(self, company_name: str) -> List[PatentData]:
        """Fetch patents for a specific company via POST with small pages + retries."""
        if self._patents_api_disabled:
            return []
        patents: List[PatentData] = []
        variations = self._generate_company_variations(company_name)
        url = self.api_url if self.api_url.endswith('/') else self.api_url + '/'

        @circuit_breaker(failure_threshold=3, recovery_timeout=60, name="patentsview_org_api")
        def _cb_post_page(payload: Dict):
            return requests.post(url, json=payload, headers={**self.headers, 'Content-Type': 'application/json'}, timeout=20)

        def post_page(page: int, per_page: int = 100, retries: int = 2):
            payload = {
                "q": {"_or": [{"assignees.assignee_organization": v} for v in variations]},
                "f": ["patent_id","patent_title","assignees"],
                "o": {"per_page": per_page, "page": page, "exclude_withdrawn": True}
            }
            backoff = 0.8
            for _ in range(retries):
                try:
                    r = _cb_post_page(payload)
                    if r.status_code == 200:
                        return r.json()
                    logger.warning(f"Org POST {company_name} page {page} failed {r.status_code}")
                except Exception as e:
                    logger.debug(f"Org POST exception {company_name} page {page}: {e}")
                time.sleep(backoff)
                backoff *= 2
            return None

        # Fetch up to 3 pages
        for page in range(1, 4):
            data = post_page(page)
            if not data:
                break
            rows = data.get('patents', [])
            if not rows:
                break
            for patent in rows:
                for assignee in (patent.get('assignees') or []):
                    org = assignee.get('assignee_organization', '')
                    if any(var.upper() in org.upper() for var in variations):
                        patents.append(PatentData(
                            patent_id=patent.get('patent_id'),
                            title=patent.get('patent_title',''),
                            assignee_org=org,
                            assignee_city=assignee.get('assignee_city',''),
                            assignee_state=assignee.get('assignee_state','')
                        ))
            time.sleep(0.5)

        if patents:
            logger.info(f"Found {len(patents)} patents for {company_name}")
        return patents
    
    def _build_patent_index(self, patents: List[PatentData]) -> Dict[str, List[PatentData]]:
        """Build searchable index of patents by organization"""
        index = defaultdict(list)
        
        for patent in patents:
            # Index by exact organization name
            org_upper = patent.assignee_org.upper()
            index[org_upper].append(patent)
            
            # Also index by cleaned name
            clean_org = self._clean_business_name(patent.assignee_org).upper()
            if clean_org != org_upper:
                index[clean_org].append(patent)
            
            # Index by first few words for partial matching
            words = clean_org.split()
            if len(words) >= 2:
                partial_key = ' '.join(words[:2])
                index[partial_key].append(patent)
        
        logger.info(f"Built patent index with {len(index)} unique organization keys")
        return dict(index)
    
    def _match_businesses_to_patents(self, 
                                   business_names: List[str], 
                                   patent_index: Dict[str, List[PatentData]]) -> Dict[str, int]:
        """Match business names to patents using various strategies"""
        patent_counts = {}
        
        for business_name in business_names:
            count = 0
            
            # Strategy 1: Exact match on full name
            business_upper = business_name.upper()
            if business_upper in patent_index:
                count += len(patent_index[business_upper])
            
            # Strategy 2: Match on cleaned name
            clean_name = self._clean_business_name(business_name).upper()
            if clean_name != business_upper and clean_name in patent_index:
                count += len(patent_index[clean_name])
            
            # Strategy 3: Partial match on first words
            words = clean_name.split()
            if len(words) >= 2:
                partial_key = ' '.join(words[:2])
                if partial_key in patent_index:
                    # Verify these are actual matches
                    for patent in patent_index[partial_key]:
                        if self._is_likely_match(clean_name, patent.assignee_org):
                            count += 1
            
            # Strategy 4: Search for business name within organization names
            if count == 0 and len(clean_name) > 5:  # Only for meaningful names
                for org_name, org_patents in patent_index.items():
                    if clean_name in org_name or org_name.startswith(clean_name):
                        # Verify match quality
                        if self._is_likely_match(clean_name, org_name):
                            count += len(org_patents)
            
            patent_counts[business_name] = count
            
            if count > 0:
                logger.debug(f"Found {count} patents for {business_name}")
        
        return patent_counts
    
    def _clean_business_name(self, name: str) -> str:
        """Clean business name for matching"""
        # Remove common suffixes
        suffixes = [
            'Inc', 'LLC', 'Corp', 'Corporation', 'Company', 'Co',
            'Ltd', 'Limited', 'Partners', 'LP', 'LLP', 'Solutions',
            'Services', 'Group', 'Associates', 'Enterprises'
        ]
        
        clean_name = name
        for suffix in suffixes:
            # Remove with various punctuation
            clean_name = re.sub(rf'\s*[,.]?\s*{suffix}\.?\s*$', '', clean_name, flags=re.IGNORECASE)
        
        # Remove trailing numbers (like #1000)
        clean_name = re.sub(r'\s*#\d+$', '', clean_name)
        
        # Remove special characters but keep spaces
        clean_name = re.sub(r'[^\w\s-]', ' ', clean_name)
        
        # Normalize whitespace
        clean_name = ' '.join(clean_name.split())
        
        return clean_name.strip()
    
    def _generate_company_variations(self, company_name: str) -> List[str]:
        """Generate variations of company name for search"""
        base_name = self._clean_business_name(company_name)
        variations = [
            base_name,
            company_name,  # Original
            f"{base_name} Corporation",
            f"{base_name} Inc",
            f"{base_name} LLC",
            f"{base_name} Company",
            f"{base_name} Technologies",
            f"{base_name} Systems"
        ]
        
        # For known companies, add specific variations
        known_variations = {
            "Cerner": ["Cerner Corporation", "Cerner Innovation"],
            "Garmin": ["Garmin Ltd", "Garmin International"],
            "Sprint": ["Sprint Corporation", "Sprint Communications"],
            "Hallmark": ["Hallmark Cards", "Hallmark Marketing"],
            "H&R Block": ["HRB Tax Group", "H&R Block Tax Services"]
        }
        
        if base_name in known_variations:
            variations.extend(known_variations[base_name])
        
        return list(set(variations))  # Remove duplicates
    
    def _is_likely_match(self, business_name: str, org_name: str) -> bool:
        """Determine if business name likely matches organization name"""
        # Both names should be cleaned and uppercase
        business_clean = business_name.upper()
        org_clean = self._clean_business_name(org_name).upper()
        
        # Exact match after cleaning
        if business_clean == org_clean:
            return True
        
        # Business name is contained in org name
        if business_clean in org_clean:
            return True
        
        # Org name starts with business name
        if org_clean.startswith(business_clean):
            return True
        
        # For short names, require exact match of first word
        business_words = business_clean.split()
        org_words = org_clean.split()
        
        if len(business_words) == 1 and len(org_words) > 0:
            return business_words[0] == org_words[0]
        
        # For longer names, check if first two words match
        if len(business_words) >= 2 and len(org_words) >= 2:
            return (business_words[0] == org_words[0] and 
                   business_words[1] == org_words[1])
        
        return False

    def get_patent_statistics(self, patent_counts: Dict[str, int]) -> Dict:
        """Generate statistics about patent distribution"""
        total_businesses = len(patent_counts)
        businesses_with_patents = sum(1 for count in patent_counts.values() if count > 0)
        total_patents = sum(patent_counts.values())
        
        # Get top patent holders
        sorted_holders = sorted(
            [(name, count) for name, count in patent_counts.items() if count > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "total_businesses": total_businesses,
            "businesses_with_patents": businesses_with_patents,
            "percentage_with_patents": (businesses_with_patents / total_businesses * 100) if total_businesses > 0 else 0,
            "total_patents": total_patents,
            "average_patents_per_holder": (total_patents / businesses_with_patents) if businesses_with_patents > 0 else 0,
            "top_10_patent_holders": sorted_holders[:10],
            "patent_distribution": {
                "1-5 patents": sum(1 for c in patent_counts.values() if 1 <= c <= 5),
                "6-20 patents": sum(1 for c in patent_counts.values() if 6 <= c <= 20),
                "21-50 patents": sum(1 for c in patent_counts.values() if 21 <= c <= 50),
                "50+ patents": sum(1 for c in patent_counts.values() if c > 50)
            }
        }
