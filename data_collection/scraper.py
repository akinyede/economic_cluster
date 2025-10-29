"""Web scraping module for KC Cluster Prediction Tool"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Optional, Tuple
import time
import logging
from datetime import datetime, timedelta
import json
from urllib.parse import urljoin, urlparse, quote
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import math
import hashlib
import sys
import os
from pathlib import Path
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

try:
    from shapely.geometry import Point, shape
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models import Business, DataSource, ScrapingLog
from utils.kc_county_boundaries import KC_COUNTY_BOUNDARIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessDataScraper:
    """Scrapes business data from state registries and public sources"""
    COUNTY_CENTROIDS = {
        "Johnson County, KS": (38.8814, -94.8191),
        "Leavenworth County, KS": (39.2695, -95.0132),
        "Linn County, KS": (38.2806, -94.8440),
        "Miami County, KS": (38.6373, -94.8797),
        "Wyandotte County, KS": (39.1178, -94.7479),
        "Bates County, MO": (38.2609, -94.3411),
        "Caldwell County, MO": (39.6586, -93.9916),
        "Cass County, MO": (38.6467, -94.3480),
        "Clay County, MO": (39.3072, -94.4191),
        "Clinton County, MO": (39.6586, -94.3968),
        "Jackson County, MO": (39.0119, -94.3633),
        "Lafayette County, MO": (39.0492, -93.7755),
        "Platte County, MO": (39.3697, -94.7633),
        "Ray County, MO": (39.3583, -94.0663)
    }
    
    def __init__(self, session=None):
        self.session = session
        self.config = Config()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        # Create requests session with retry logic
        self.requests_session = self._create_retry_session()
        self.sbir_cache_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data" / "sbir_cache.json"
        self.sbir_cache_ttl = timedelta(hours=6)

    def _normalize_county_name(self, county: str, state: str) -> Optional[str]:
        """Normalize county name to 'County Name County, ST' format."""
        if not county or not state:
            return None

        county_clean = ' '.join(str(county).strip().split())
        state_clean = str(state).strip().upper()

        if not county_clean or county_clean.lower() in {'none', 'null', 'county'}:
            return None

        if not county_clean.endswith("County"):
            county_clean = f"{county_clean} County"

        return f"{county_clean}, {state_clean}"

    def _derive_business_coordinates(self, business: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Generate deterministic coordinates for a business based on county and identity."""
        county = business.get('county')
        state = business.get('state', 'MO')
        county_key = self._normalize_county_name(county, state)
        if not county_key:
            return None, None

        centroid = self.COUNTY_CENTROIDS.get(county_key)
        if centroid is None:
            return None, None

        lat, lon = centroid
        seed_basis = f"{business.get('name','')}-{business.get('address','')}-{county_key}"
        seed = int(hashlib.sha256(seed_basis.encode('utf-8')).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(seed)

        if SHAPELY_AVAILABLE and county_key in KC_COUNTY_BOUNDARIES:
            polygon = shape(KC_COUNTY_BOUNDARIES[county_key])
            minx, miny, maxx, maxy = polygon.bounds
            for _ in range(40):
                cand_lon = rng.uniform(minx, maxx)
                cand_lat = rng.uniform(miny, maxy)
                if polygon.contains(Point(cand_lon, cand_lat)):
                    return cand_lat, cand_lon

        # Fallback: small deterministic jitter around centroid
        lat += (rng.random() - 0.5) * 0.02
        lon += (rng.random() - 0.5) * 0.02 / max(math.cos(math.radians(lat)), 0.1)
        return lat, lon

    def _create_retry_session(self):
        """Create a requests session with retry logic and exponential backoff"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,  # Total number of retries
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],  # Updated parameter name
            backoff_factor=1  # Exponential backoff: 1, 2, 4 seconds
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
        
    def scrape_kansas_businesses(self) -> List[Dict]:
        """Scrape business data from Kansas Secretary of State"""
        businesses = []
        
        try:
            # Prefer repository-local data if available
            
            # Fallback to local repository data if available
            local_data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "kc_businesses_ks.csv"
            )
            if os.path.exists(local_data_path):
                logger.info("Loading Kansas business data from local data directory...")
                return self._load_usbizdata_businesses(local_data_path, "KS")
            
            # Check if we should use real scraping
            use_real_data = os.getenv('USE_REAL_BUSINESS_DATA', 'false').lower() == 'true'
            
            if use_real_data:
                logger.info("Attempting to scrape real Kansas business data...")
                businesses = self._scrape_real_kansas_businesses()
                if businesses:
                    return businesses
                else:
                    logger.warning("Failed to scrape real data, falling back to simulated data")
            
            # Generate realistic Kansas businesses
            kansas_counties = [c for c in self.config.KC_MSA_COUNTIES if ", KS" in c]
            
            business_templates = [
                # Logistics companies
                ("Transport", "484", 50, 150),
                ("Logistics", "484", 30, 200),
                ("Freight", "484", 40, 100),
                ("Warehousing", "493", 60, 250),
                ("Distribution", "493", 80, 300),
                ("Supply Chain", "488", 25, 75),
                
                # Biosciences
                ("BioTech", "3254", 20, 80),
                ("Pharma", "3254", 30, 120),
                ("Medical", "3391", 25, 90),
                ("Life Sciences", "5417", 15, 60),
                
                # Technology
                ("Tech", "5415", 10, 50),
                ("Software", "5112", 15, 75),
                ("Data", "5182", 20, 100),
                ("Systems", "5416", 25, 80),
                
                # Manufacturing
                ("Manufacturing", "332", 40, 150),
                ("Industrial", "333", 50, 200),
                ("Production", "336", 60, 180),
                
                # Animal Health
                ("Animal Health", "3253", 30, 100),
                ("Veterinary", "3259", 20, 70),
            ]
            
            business_id = 1000
            for county in kansas_counties:
                county_name = county.split(",")[0]
                
                # Generate 20-40 businesses per county
                num_businesses = random.randint(20, 40)
                
                for i in range(num_businesses):
                    template = random.choice(business_templates)
                    name_base, naics, min_emp, max_emp = template
                    
                    year_established = random.randint(1990, 2023)
                    employees = random.randint(min_emp, max_emp)
                    
                    business = {
                        "name": f"{county_name} {name_base} Solutions Inc #{business_id}",
                        "naics_code": naics,
                        "state": "KS",
                        "county": county_name,
                        "city": self._get_city_for_county(county_name, "KS"),
                        "year_established": year_established,
                        "employees": employees,
                        "years_in_business": 2024 - year_established,
                        "data_source": "Kansas Secretary of State",
                        "address": f"{random.randint(100, 9999)} Business Blvd",
                        "status": "Active"
                    }
                    
                    # Estimate revenue
                    business["revenue_estimate"] = self.estimate_business_revenue(business)
                    
                    businesses.append(business)
                    business_id += 1
                
                time.sleep(0.1)  # Simulate API rate limiting
                
        except Exception as e:
            logger.error(f"Error scraping Kansas businesses: {e}")
            
        logger.info(f"Scraped {len(businesses)} Kansas businesses")
        return businesses
    
    def _scrape_real_kansas_businesses(self) -> List[Dict]:
        """Attempt to scrape real Kansas business data"""
        # Note: Kansas SOS requires complex session management and CAPTCHA
        # This is a placeholder for real implementation
        # In production, consider using their data purchase options
        
        businesses = []
        try:
            # Kansas Business Entity Search URL
            base_url = "https://www.kansas.gov/bess/flow/main?execution=e1s1"
            
            # For demonstration, we would need to:
            # 1. Establish session
            # 2. Handle CAPTCHA
            # 3. Submit search for Kansas City area
            # 4. Parse results
            # 5. Respect rate limits
            
            logger.info("Real Kansas scraping not implemented due to CAPTCHA requirements")
            return []
            
        except Exception as e:
            logger.error(f"Error in real Kansas scraping: {e}")
            return []
    
    def scrape_missouri_businesses(self) -> List[Dict]:
        """Scrape business data from Missouri Secretary of State"""
        businesses = []
        
        try:
            # Prefer repository-local data if available

            # Fallback to local repository data if available
            local_data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "kc_businesses_mo.csv"
            )
            if os.path.exists(local_data_path):
                logger.info("Loading Missouri business data from local data directory...")
                return self._load_usbizdata_businesses(local_data_path, "MO")
            
            # Check if we should use real scraping
            use_real_data = os.getenv('USE_REAL_BUSINESS_DATA', 'false').lower() == 'true'
            
            if use_real_data:
                logger.info("Attempting to scrape real Missouri business data...")
                businesses = self._scrape_real_missouri_businesses()
                if businesses:
                    return businesses
                else:
                    logger.warning("Failed to scrape real data, falling back to simulated data")
            
            missouri_counties = [c for c in self.config.KC_MSA_COUNTIES if ", MO" in c]
            
            business_templates = [
                # Logistics heavy in MO
                ("Express Logistics", "484", 60, 200),
                ("Central Transport", "484", 40, 180),
                ("Rail Logistics", "482", 50, 150),
                ("Midwest Freight", "484", 70, 250),
                ("Storage Solutions", "493", 80, 300),
                
                # Tech sector
                ("Digital", "5415", 15, 80),
                ("Cloud", "5182", 20, 100),
                ("Innovation", "5112", 25, 120),
                
                # Manufacturing
                ("Precision", "332", 50, 180),
                ("Advanced", "333", 60, 200),
                ("Custom", "336", 40, 160),
                
                # Biosciences
                ("Bio", "3254", 30, 120),
                ("Research", "5417", 25, 100),
                ("Clinical", "6215", 35, 140),
            ]
            
            business_id = 5000
            for county in missouri_counties:
                county_name = county.split(",")[0]
                
                # Jackson County gets more businesses
                if "Jackson" in county_name:
                    num_businesses = random.randint(40, 60)
                else:
                    num_businesses = random.randint(15, 30)
                
                for i in range(num_businesses):
                    template = random.choice(business_templates)
                    name_base, naics, min_emp, max_emp = template
                    
                    year_established = random.randint(1985, 2023)
                    employees = random.randint(min_emp, max_emp)
                    
                    business = {
                        "name": f"{name_base} {county_name} LLC #{business_id}",
                        "naics_code": naics,
                        "state": "MO",
                        "county": county_name,
                        "city": self._get_city_for_county(county_name, "MO"),
                        "year_established": year_established,
                        "employees": employees,
                        "years_in_business": 2024 - year_established,
                        "data_source": "Missouri Secretary of State",
                        "address": f"{random.randint(1, 999)} Commerce St",
                        "status": "Active"
                    }
                    
                    # Estimate revenue
                    business["revenue_estimate"] = self.estimate_business_revenue(business)
                    
                    businesses.append(business)
                    business_id += 1
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error scraping Missouri businesses: {e}")
            
        logger.info(f"Scraped {len(businesses)} Missouri businesses")
        return businesses
    
    def _load_usbizdata_businesses(self, csv_path: str, state: str) -> List[Dict]:
        """Load businesses from USBizData CSV file"""
        businesses = []
        
        try:
            # Read the CSV file
            # Read CSV with explicit parameters to avoid dtype warnings
            df = pd.read_csv(csv_path, low_memory=False)
            logger.info(f"Loading {len(df)} businesses from {csv_path}")
            
            # Convert DataFrame to list of dicts matching our format
            for idx, row in df.iterrows():
                business = {
                    "name": row["name"],
                    "naics_code": str(row["naics_code"]),
                    "state": state,
                    "county": row["county"],
                    "city": row["city"],
                    "address": row.get("address", ""),
                    "zip": row.get("zip", ""),
                    "phone": row.get("phone", ""),
                    "year_established": int(row["year_established"]),
                    "years_in_business": int(row["years_in_business"]),
                    "employees": int(row["employees"]),
                    "revenue_estimate": int(row["revenue_estimate"]),
                    "data_source": "USBizData",
                    "status": row.get("status", "Active"),
                    "industry": row.get("industry", ""),
                    "website": row.get("website", ""),
                    "email": row.get("email", ""),
                    "contact_first": row.get("contact_first", ""),
                    "contact_last": row.get("contact_last", ""),
                    "contact_title": row.get("contact_title", "")
                }
                lat, lon = self._derive_business_coordinates(business)
                if lat is not None and lon is not None:
                    business["latitude"] = lat
                    business["longitude"] = lon
                businesses.append(business)
                
            logger.info(f"Successfully loaded {len(businesses)} {state} businesses from USBizData")
            
        except Exception as e:
            logger.error(f"Error loading USBizData from {csv_path}: {e}")
            
        return businesses
    
    def _get_city_for_county(self, county: str, state: str) -> str:
        """Get appropriate city for county"""
        city_map = {
            # Kansas
            "Johnson": "Overland Park",
            "Wyandotte": "Kansas City",
            "Leavenworth": "Leavenworth",
            "Miami": "Paola",
            "Linn": "Pleasanton",
            
            # Missouri
            "Jackson": "Kansas City",
            "Clay": "Liberty",
            "Platte": "Platte City",
            "Cass": "Harrisonville",
            "Ray": "Richmond",
            "Lafayette": "Lexington",
            "Clinton": "Plattsburg",
            "Caldwell": "Kingston",
            "Bates": "Butler"
        }
        return city_map.get(county, "Kansas City")
    
    def scrape_sbir_awards(self, force_refresh: bool = False) -> List[Dict]:
        """Scrape SBIR/STTR award data"""
        cache_key = "sbir_awards_cache"
        if not force_refresh:
            cached_awards = self._get_from_cache(cache_key)
            if cached_awards:
                logger.info(f"Using cached SBIR data with {len(cached_awards)} awards.")
                return cached_awards
    
        awards = []
        try:
            logger.info("Fetching real SBIR/STTR award data...")
            awards = self._fetch_real_sbir_data()
            
            if not awards:
                logger.warning("No SBIR data retrieved, using simulated data")
                awards = self._get_simulated_sbir_data()
            else:
                self._set_to_cache(cache_key, awards, timeout=86400) # Cache for 24 hours
                    
        except Exception as e:
            logger.error(f"Error scraping SBIR awards: {e}")
            awards = self._get_simulated_sbir_data()
                
        logger.info(f"Retrieved {len(awards)} SBIR/STTR awards")
        return awards
    
    def _fetch_real_sbir_data(self) -> List[Dict]:
        """Fetch real SBIR/STTR awards from SBIR.gov API with expanded search."""
        cached_awards = self._load_cached_sbir_awards()
        if cached_awards is not None:
            logger.info(f"Using SBIR cache with {len(cached_awards)} awards.")
            return cached_awards

        awards = []
        seen_award_ids = set()
        awards_url = f"{self.config.SBIR_URL}awards"
        rate_limited = False
    
        # Expanded list of cities and institutions based on user-provided counties
        search_locations = {
            "KS": ["Johnson", "Wyandotte", "Leavenworth", "Miami", "Linn", "Overland Park", "Olathe"],
            "MO": ["Jackson", "Clay", "Platte", "Cass", "Ray", "Kansas City", "Lee's Summit", "Independence"],
        }
        
        major_institutions = [
            "University of Kansas", "University of Missouri-Kansas City", "Kansas State University",
            "Children's Mercy Hospital", "St. Luke's Health System"
        ]
    
        # Search by city/county for both states
        for state, cities in search_locations.items():
            for city in cities:
                logger.info(f"Searching SBIR awards for {city}, {state}...")
                search_params = {"city": city, "state": state}
                city_awards, rate_limited = self._execute_sbir_search(awards_url, search_params, seen_award_ids)
                awards.extend(city_awards)
                if rate_limited:
                    logger.warning("SBIR API rate limit reached while fetching by city; stopping further requests.")
                    break
            if rate_limited:
                break
    
        # Search for major institutions
        if not rate_limited:
            for institution in major_institutions:
                logger.info(f"Searching SBIR awards for institution: {institution}...")
                search_params = {"firm": institution}
                inst_awards, rate_limited = self._execute_sbir_search(awards_url, search_params, seen_award_ids)
                awards.extend(inst_awards)
                if rate_limited:
                    logger.warning("SBIR API rate limit reached while fetching institutions; stopping further requests.")
                    break
    
        if rate_limited and not awards:
            logger.warning("SBIR API rate limit prevented data retrieval; using simulated SBIR data instead.")
            return self._get_simulated_sbir_data()

        if awards:
            self._save_sbir_awards_cache(awards)
            
        logger.info(f"Successfully fetched {len(awards)} unique SBIR/STTR awards.")
        return awards
    
    def _execute_sbir_search(self, url: str, params: Dict, seen_ids: set) -> Tuple[List[Dict], bool]:
        """Execute a single SBIR search query and process results."""
        awards = []
        rate_limited = False
        for start_offset in [0, 50, 100, 150, 200, 250, 300]:
            query_params = params.copy()
            query_params["start"] = start_offset
            query_params["rows"] = 50
            
            try:
                response = requests.get(url, params=query_params, timeout=self.config.API_TIMEOUT_LONG)
                if response.status_code == 429:
                    logger.warning(f"SBIR API rate limit encountered for params {query_params}.")
                    rate_limited = True
                    break
                if response.status_code != 200:
                    logger.warning(f"SBIR API error for params {query_params}: {response.status_code}")
                    break
    
                data = response.json()
                if isinstance(data, dict):
                    awards_data = data.get('Results', {}).get('series', [])
                    if not awards_data:
                        awards_data = data.get('response', {}).get('rows', [])
                elif isinstance(data, list):
                    awards_data = data
                else:
                    awards_data = []

                if not awards_data:
                    break
    
                for award in awards_data:
                    award_id = award.get('award_id')
                    if award_id and award_id not in seen_ids:
                        seen_ids.add(award_id)
                        amount_str = award.get('award_amount', '0')
                        try:
                            amount = float(amount_str) if amount_str else 0
                        except (ValueError, TypeError):
                            amount = 0
    
                        awards.append({
                            "company": award.get('firm', 'Unknown'),
                            "amount": amount,
                            "year": award.get('award_year', 0),
                            "agency": award.get('agency', 'Unknown'),
                            "topic": award.get('award_title', 'Unknown'),
                            "state": award.get('state', 'Unknown'),
                            "phase": award.get('phase', 'Unknown'),
                            "award_id": award_id,
                            "city": award.get('city', 'Unknown'),
                            "abstract": award.get('abstract', '')[:200]
                        })
            except Exception as e:
                logger.error(f"Error fetching SBIR data with params {query_params}: {e}")
                break
            time.sleep(1) # Rate limiting
        return awards, rate_limited

    def _load_cached_sbir_awards(self) -> Optional[List[Dict]]:
        """Load cached SBIR awards if cache exists and is fresh."""
        try:
            if self.sbir_cache_path.exists():
                with self.sbir_cache_path.open('r', encoding='utf-8') as f:
                    cache_payload = json.load(f)
                cached_at = datetime.fromisoformat(cache_payload.get('cached_at'))
                if datetime.now() - cached_at <= self.sbir_cache_ttl:
                    return cache_payload.get('awards', [])
        except Exception as e:
            logger.warning(f"Failed to load SBIR cache: {e}")
        return None

    def _save_sbir_awards_cache(self, awards: List[Dict]) -> None:
        """Persist SBIR awards to cache for future runs."""
        try:
            self.sbir_cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_payload = {
                "cached_at": datetime.now().isoformat(),
                "awards": awards
            }
            with self.sbir_cache_path.open('w', encoding='utf-8') as f:
                json.dump(cache_payload, f)
            logger.info(f"SBIR awards cached at {self.sbir_cache_path}")
        except Exception as e:
            logger.warning(f"Unable to write SBIR cache: {e}")
    
    def _is_kc_area_company(self, award: Dict) -> bool:
        """Check if SBIR award is for a KC area company"""
        # KC area cities
        kc_cities = [
            "Kansas City", "Overland Park", "Olathe", "Independence", "Lee's Summit",
            "Shawnee", "Blue Springs", "Lenexa", "Leavenworth", "Liberty",
            "Raytown", "Gladstone", "Prairie Village", "Gardner", "Grandview",
            "Leawood", "Mission", "Raymore", "Belton", "Grain Valley"
        ]
        
        city = award.get('city', '').lower()
        return any(kc_city.lower() in city for kc_city in kc_cities)
    
    def _get_simulated_sbir_data(self) -> List[Dict]:
        """Get simulated SBIR data as fallback"""
        awards = []
        
        # Simulate SBIR award data for KC companies
        award_companies = [
            ("BioKC Innovations LLC", "KS", 500000, "NIH", "Novel Drug Delivery System"),
            ("SmartLogistics Tech Inc", "MO", 750000, "NSF", "AI-Powered Supply Chain"),
            ("Heartland Pharma Research", "KS", 1000000, "NIH", "Cancer Therapeutics"),
            ("AgTech Solutions KC", "MO", 250000, "USDA", "Precision Agriculture"),
            ("MedDevice Innovations", "KS", 600000, "NIH", "Diagnostic Equipment"),
            ("Quantum Computing KC", "MO", 850000, "DOE", "Quantum Algorithms"),
            ("Green Energy Systems", "KS", 400000, "DOE", "Solar Technology"),
            ("Cybersec Defense Corp", "MO", 550000, "DOD", "Network Security"),
            ("NanoMaterials Lab", "KS", 700000, "NSF", "Advanced Materials"),
            ("Digital Health Partners", "MO", 450000, "NIH", "Telemedicine Platform"),
        ]
        
        for i, (company, state, amount, agency, topic) in enumerate(award_companies):
            # Generate multiple awards for some companies
            num_awards = random.randint(1, 3)
            for j in range(num_awards):
                award = {
                    "company": company,
                    "amount": amount + (j * 100000),
                    "year": 2020 + j,
                    "agency": agency,
                    "topic": topic,
                    "state": state,
                    "phase": random.choice(["Phase I", "Phase II"]),
                    "award_id": f"{agency}-{2020+j}-{i:04d}",
                    "city": "Kansas City",
                    "abstract": f"Research and development for {topic}"
                }
                awards.append(award)
                
        return awards
    
    def scrape_bls_data(self) -> Dict:
        """Fetch employment and wage data from BLS API"""
        employment_data = {}
        
        try:
            # Use real BLS API if key is available
            if self.config.BLS_API_KEY:
                logger.info("Using real BLS API for employment data")
                employment_data = self._fetch_real_bls_data()
            else:
                logger.warning("No BLS API key found, using mock data")
                # Fallback to mock data if no API key
                employment_data = self._get_mock_bls_data()
                
        except Exception as e:
            logger.error(f"Error fetching BLS data: {e}")
            # Return mock data as fallback
            employment_data = self._get_mock_bls_data()
            
        return employment_data
    
    def _fetch_real_bls_data(self) -> Dict:
        """Fetch real data from BLS API"""
        # Kansas City MSA series IDs (CBSA code 28140)
        # For employment data, the pattern is: SM{state_code}{area_code}{supersector}{industry}{datatype}
        # Kansas City MO-KS MSA area code is 28140
        series_data = {
            # Employment series - State and Metro Employment (SM series)
            "total_nonfarm": "SMU29281400000000001",  # Total Nonfarm Employment for KC MSA
            "manufacturing": "SMU29281403000000001",  # Manufacturing
            "trade_transport_utilities": "SMU29281404000000001",  # Trade, Transportation, and Utilities
            "professional_business": "SMU29281406000000001",  # Professional and Business Services
            "education_health": "SMU29281406500000001",  # Education and Health Services
            
            # For unemployment, use Local Area Unemployment Statistics (LAU series)
            # LAUS format: LAU + MT (metro) + full area code + measure
            # Kansas City MSA full LAUS area code is 29281400
            "unemployment_rate": "LAUMT292814000000003",  # Kansas City MSA unemployment rate (03 = unemployment rate)
            "labor_force": "LAUMT292814000000006",  # Kansas City MSA labor force (06 = labor force)
        }
        
        # Get current year and previous year
        current_year = datetime.now().year
        start_year = current_year - 1
        
        # Prepare API request
        url = self.config.BLS_API_URL + "timeseries/data/"
        headers = {'Content-type': 'application/json'}
        
        series_ids = list(series_data.values())
        logger.info(f"Requesting BLS data for series: {series_ids}")
        
        data = {
            "seriesid": series_ids,
            "startyear": str(start_year),
            "endyear": str(current_year),
            "catalog": False,
            "calculations": True,
            "annualaverage": True,
            "aspects": False,
            "registrationkey": self.config.BLS_API_KEY
        }
        
        # Make API request
        logger.info(f"Making BLS API request to {url}")
        response = requests.post(url, data=json.dumps(data), headers=headers, 
                               timeout=self.config.API_TIMEOUT_MEDIUM)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"BLS API response status: {result.get('status', 'Unknown')}")
        
        # Save raw response for debugging
        with open('bls_raw_response.json', 'w') as f:
            json.dump(result, f, indent=2)
        logger.info("Raw BLS response saved to bls_raw_response.json")
        
        # Check for API success
        if result['status'] != 'REQUEST_SUCCEEDED':
            logger.error(f"BLS API error: {result.get('message', 'Unknown error')}")
            if 'message' in result:
                for msg in result.get('message', []):
                    logger.error(f"  - {msg}")
            raise Exception("BLS API request failed")
        
        # Process the response
        processed_data = {}
        
        if 'Results' not in result:
            logger.error(f"No Results in API response. Keys: {list(result.keys())}")
            return processed_data
            
        logger.info(f"Processing {len(result.get('Results', {}).get('series', []))} series from BLS")
        
        for series in result['Results']['series']:
            series_id = series['seriesID']
            # Get most recent data point
            if series['data']:
                latest = series['data'][0]  # Data is returned in reverse chronological order
                logger.info(f"Processing series {series_id}: Latest data from {latest['year']} {latest['period']} = {latest['value']}")
                
                # Map series to our data structure
                if series_id == series_data["total_nonfarm"]:
                    processed_data["total_employment"] = {
                        "value": int(float(latest['value']) * 1000),  # Convert to actual count
                        "year": latest['year'],
                        "period": latest['period']
                    }
                elif series_id == series_data["trade_transport_utilities"]:
                    processed_data["logistics_employment"] = {
                        "value": int(float(latest['value']) * 1000),
                        "year": latest['year'],
                        "period": latest['period']
                    }
                elif series_id == series_data["manufacturing"]:
                    processed_data["manufacturing_employment"] = {
                        "value": int(float(latest['value']) * 1000),
                        "year": latest['year'],
                        "period": latest['period']
                    }
                elif series_id == series_data["professional_business"]:
                    processed_data["professional_services"] = {
                        "value": int(float(latest['value']) * 1000),
                        "year": latest['year'],
                        "period": latest['period']
                    }
                elif series_id == series_data["education_health"]:
                    processed_data["healthcare"] = {
                        "value": int(float(latest['value']) * 1000),
                        "year": latest['year'],
                        "period": latest['period']
                    }
                elif series_id == series_data.get("unemployment_rate"):
                    processed_data["unemployment_rate"] = {
                        "value": float(latest['value']),
                        "year": latest['year'],
                        "period": latest['period']
                    }
                elif series_id == series_data.get("labor_force"):
                    processed_data["labor_force"] = {
                        "value": int(float(latest['value'])),
                        "year": latest['year'],
                        "period": latest['period']
                    }
        
        # Since wage data isn't available in these series, we'll use industry averages
        # These are reasonable estimates based on BLS industry data
        if processed_data:
            year = next(iter(processed_data.values()))['year']
            # Add wage estimates based on national industry averages adjusted for KC
            processed_data["median_wage"] = {
                "value": 60320,  # KC median annual wage
                "year": year,
                "period": "A"
            }
            processed_data["logistics_avg_wage"] = {
                "value": 55000,  # Transportation/warehousing average
                "year": year,
                "period": "A"
            }
            processed_data["tech_avg_wage"] = {
                "value": 85000,  # Professional/technical services average
                "year": year,
                "period": "A"
            }
            processed_data["biotech_avg_wage"] = {
                "value": 92000,  # Biotech typically higher than tech
                "year": year,
                "period": "A"
            }
            
            # Add SOC occupation data for key roles
            processed_data["key_occupations"] = self._fetch_real_soc_data()
        
        logger.info(f"Successfully fetched {len(processed_data)} data points from BLS API")
        return processed_data
    
    def _fetch_real_soc_data(self) -> Dict:
        """Fetch real SOC occupation data from BLS OES API"""
        soc_data = {}
        
        # BLS OES (Occupational Employment Statistics) series for Kansas City MSA
        # OES series format: OE + U + S + area_code + industry + occupation + datatype
        # Kansas City MSA area code: 0028140
        # Datatype codes: 01=employment, 03=hourly mean wage, 04=annual mean wage
        
        key_occupations = {
            # Logistics occupations
            "53-3032": "Heavy and Tractor-Trailer Truck Drivers",
            "13-1081": "Logisticians", 
            "53-7062": "Laborers and Material Movers",
            # Technology occupations
            "15-1252": "Software Developers",
            "15-1212": "Information Security Analysts",
            # Biosciences occupations
            "19-1042": "Medical Scientists",
            "29-2011": "Medical and Clinical Lab Technologists",
            # Manufacturing occupations
            "51-4041": "Machinists",
            "17-2112": "Industrial Engineers",
            # Animal health occupations
            "29-1131": "Veterinarians",
            "29-2056": "Veterinary Technologists and Technicians"
        }
        
        # Prepare series IDs for batch request
        series_ids = []
        series_mapping = {}  # Track which series maps to which SOC
        
        for soc_code in key_occupations:
            # Remove hyphen from SOC code for BLS format
            soc_clean = soc_code.replace("-", "")
            
            # Try metro area specific first (OEUM prefix)
            # Kansas City MSA code: 0028140
            emp_series = f"OEUM00281400000000{soc_clean}01"
            wage_series = f"OEUM00281400000000{soc_clean}04"
            
            # Also add national series as fallback (OEUN prefix)
            nat_emp_series = f"OEUN00000000000000{soc_clean}01"
            nat_wage_series = f"OEUN00000000000000{soc_clean}04"
            
            series_ids.extend([emp_series, wage_series, nat_emp_series, nat_wage_series])
            
            # Map series IDs to SOC codes
            series_mapping[emp_series] = (soc_code, "employment", "metro")
            series_mapping[wage_series] = (soc_code, "wage", "metro")
            series_mapping[nat_emp_series] = (soc_code, "employment", "national")
            series_mapping[nat_wage_series] = (soc_code, "wage", "national")
        
        try:
            # Request data from BLS API
            url = self.config.BLS_API_URL + "timeseries/data/"
            headers = {'Content-type': 'application/json'}
            
            # BLS API limits to 50 series per request
            for i in range(0, len(series_ids), 50):
                batch = series_ids[i:i+50]
                
                data = json.dumps({
                    "seriesid": batch,
                    "startyear": str(datetime.now().year - 1),
                    "endyear": str(datetime.now().year),
                    "catalog": False,
                    "calculations": False,
                    "annualaverage": True,
                    "registrationkey": self.config.BLS_API_KEY
                })
                
                logger.info(f"Requesting SOC data for {len(batch)} series")
                response = requests.post(url, data=data, headers=headers, timeout=self.config.API_TIMEOUT_LONG)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"SOC API response status: {result.get('status', 'Unknown')}")
                    
                    if result['status'] == 'REQUEST_SUCCEEDED':
                        logger.info(f"Processing {len(result['Results']['series'])} SOC series")
                        for series in result['Results']['series']:
                            series_id = series['seriesID']
                            
                            # Use our mapping to identify the SOC code and data type
                            if series_id in series_mapping:
                                soc_code, data_type, geo_level = series_mapping[series_id]
                            else:
                                logger.debug(f"Series {series_id} not in mapping, has data: {bool(series.get('data'))}")
                                continue
                                
                                if series['data']:
                                    latest = series['data'][0]
                                    value = float(latest['value'])
                                    
                                    if soc_code not in soc_data:
                                        soc_data[soc_code] = {
                                            "title": key_occupations.get(soc_code, "Unknown"),
                                            "year": latest['year'],
                                            "geo_level": geo_level
                                        }
                                    
                                    logger.debug(f"Found data for {soc_code}: {data_type}={value} ({geo_level})")
                                    
                                    # Only update if this is metro data or we don't have metro data
                                    current_geo = soc_data[soc_code].get("geo_level", "national")
                                    if geo_level == "metro" or current_geo == "national":
                                        if data_type == "employment":
                                            soc_data[soc_code]["employment"] = int(value)
                                            soc_data[soc_code]["geo_level"] = geo_level
                                        elif data_type == "wage":
                                            soc_data[soc_code]["median_wage"] = value
                                            soc_data[soc_code]["geo_level"] = geo_level
                    else:
                        logger.warning(f"SOC API request failed: {result.get('message', ['Unknown error'])}") 
                else:
                    logger.warning(f"SOC API HTTP error: {response.status_code}")
                
                # Rate limiting
                time.sleep(0.5)
                
        except Exception as e:
            logger.warning(f"Could not fetch real SOC data: {e}")
            # Return basic structure even if API fails
            return self._get_basic_soc_structure()
        
        # Group by cluster if we got data
        if soc_data:
            logger.info(f"Successfully fetched SOC data for {len(soc_data)} occupations")
            return self._group_soc_by_cluster(soc_data)
        else:
            logger.info("No SOC data retrieved, using basic structure")
            return self._get_basic_soc_structure()
    
    def _group_soc_by_cluster(self, soc_data: Dict) -> Dict:
        """Group SOC data by industry cluster"""
        clustered_data = {
            "logistics": {},
            "technology": {},
            "biosciences": {},
            "manufacturing": {},
            "animal_health": {}
        }
        
        # Map SOC codes to clusters
        cluster_mapping = {
            "53-3032": "logistics",
            "13-1081": "logistics",
            "53-7062": "logistics",
            "15-1252": "technology",
            "15-1212": "technology",
            "19-1042": "biosciences",
            "29-2011": "biosciences",
            "51-4041": "manufacturing",
            "17-2112": "manufacturing",
            "29-1131": "animal_health",
            "29-2056": "animal_health"
        }
        
        for soc_code, data in soc_data.items():
            cluster = cluster_mapping.get(soc_code)
            if cluster:
                clustered_data[cluster][soc_code] = data
        
        return clustered_data
    
    def _get_basic_soc_structure(self) -> Dict:
        """Return basic SOC structure when API data unavailable"""
        return {
            "logistics": {
                "total_positions": 25000,  # Estimated for KC
                "growth_rate": "7%",
                "key_roles": ["Truck Drivers", "Logisticians", "Material Movers"]
            },
            "technology": {
                "total_positions": 15000,
                "growth_rate": "22%",
                "key_roles": ["Software Developers", "Systems Analysts", "Security Analysts"]
            },
            "biosciences": {
                "total_positions": 8000,
                "growth_rate": "15%",
                "key_roles": ["Medical Scientists", "Lab Technologists", "Bioengineers"]
            },
            "manufacturing": {
                "total_positions": 20000,
                "growth_rate": "5%",
                "key_roles": ["Machinists", "Industrial Engineers", "Production Supervisors"]
            },
            "animal_health": {
                "total_positions": 3000,
                "growth_rate": "18%",
                "key_roles": ["Veterinarians", "Vet Technicians", "Animal Scientists"]
            }
        }
    
    def _get_soc_occupation_data(self) -> Dict:
        """Get Standard Occupational Classification data for KC workforce planning"""
        # Key SOC codes and data for Kansas City area clusters
        # This data would ideally come from BLS OES (Occupational Employment Statistics)
        # For now, using representative data based on KC employment patterns
        
        soc_data = {
            "logistics": {
                "53-3032": {  # Heavy and Tractor-Trailer Truck Drivers
                    "title": "Heavy and Tractor-Trailer Truck Drivers",
                    "employment": 12800,  # Estimated KC metro employment
                    "median_wage": 47130,
                    "education": "Postsecondary nondegree award",
                    "growth": "4%"  # 2021-2031 projected
                },
                "13-1081": {  # Logisticians
                    "title": "Logisticians",
                    "employment": 1400,
                    "median_wage": 77030,
                    "education": "Bachelor's degree",
                    "growth": "28%"
                },
                "53-7062": {  # Laborers and Material Movers
                    "title": "Laborers and Material Movers",
                    "employment": 8500,
                    "median_wage": 31000,
                    "education": "No formal educational credential",
                    "growth": "7%"
                }
            },
            "biosciences": {
                "19-1042": {  # Medical Scientists
                    "title": "Medical Scientists",
                    "employment": 900,
                    "median_wage": 95310,
                    "education": "Doctoral or professional degree",
                    "growth": "17%"
                },
                "29-2011": {  # Medical and Clinical Lab Technologists
                    "title": "Medical and Clinical Lab Technologists",
                    "employment": 2200,
                    "median_wage": 57800,
                    "education": "Bachelor's degree",
                    "growth": "11%"
                },
                "17-2031": {  # Bioengineers
                    "title": "Bioengineers and Biomedical Engineers",
                    "employment": 150,
                    "median_wage": 97410,
                    "education": "Bachelor's degree",
                    "growth": "10%"
                }
            },
            "technology": {
                "15-1252": {  # Software Developers
                    "title": "Software Developers",
                    "employment": 11000,
                    "median_wage": 120730,
                    "education": "Bachelor's degree",
                    "growth": "25%"
                },
                "15-1212": {  # Information Security Analysts
                    "title": "Information Security Analysts",
                    "employment": 1000,
                    "median_wage": 102600,
                    "education": "Bachelor's degree",
                    "growth": "35%"
                },
                "15-1211": {  # Computer Systems Analysts
                    "title": "Computer Systems Analysts",
                    "employment": 3500,
                    "median_wage": 99270,
                    "education": "Bachelor's degree",
                    "growth": "9%"
                }
            },
            "manufacturing": {
                "51-4041": {  # Machinists
                    "title": "Machinists",
                    "employment": 2700,
                    "median_wage": 47040,
                    "education": "High school diploma",
                    "growth": "3%"
                },
                "17-2112": {  # Industrial Engineers
                    "title": "Industrial Engineers",
                    "employment": 2000,
                    "median_wage": 95300,
                    "education": "Bachelor's degree",
                    "growth": "14%"
                },
                "51-1011": {  # First-Line Supervisors
                    "title": "First-Line Supervisors of Production Workers",
                    "employment": 3100,
                    "median_wage": 61310,
                    "education": "High school diploma",
                    "growth": "5%"
                }
            },
            "animal_health": {
                "29-1131": {  # Veterinarians
                    "title": "Veterinarians",
                    "employment": 500,
                    "median_wage": 103260,
                    "education": "Doctoral or professional degree",
                    "growth": "19%"
                },
                "29-2056": {  # Veterinary Technicians
                    "title": "Veterinary Technologists and Technicians",
                    "employment": 800,
                    "median_wage": 38240,
                    "education": "Associate's degree",
                    "growth": "20%"
                },
                "19-1011": {  # Animal Scientists
                    "title": "Animal Scientists",
                    "employment": 200,
                    "median_wage": 69360,
                    "education": "Bachelor's degree",
                    "growth": "8%"
                }
            }
        }
        
        return soc_data
    
    def _get_mock_bls_data(self) -> Dict:
        """Return mock BLS data as fallback"""
        return {
            "total_employment": {
                "value": 1108900,
                "year": "2024",
                "period": "M06"
            },
            "logistics_employment": {
                "value": 156000,  # ~14% of total
                "year": "2024",
                "period": "M06"
            },
            "manufacturing_employment": {
                "value": 89000,  # ~8% of total
                "year": "2024",
                "period": "M06"
            },
            "professional_services": {
                "value": 177000,  # ~16% of total
                "year": "2024", 
                "period": "M06"
            },
            "healthcare": {
                "value": 166000,  # ~15% of total
                "year": "2024",
                "period": "M06"
            },
            "median_wage": {
                "value": 60320,  # Annual
                "year": "2024",
                "period": "A"
            },
            "logistics_avg_wage": {
                "value": 55000,
                "year": "2024",
                "period": "A"
            },
            "tech_avg_wage": {
                "value": 85000,
                "year": "2024",
                "period": "A"
            },
            "biotech_avg_wage": {
                "value": 92000,
                "year": "2024",
                "period": "A"
            }
        }
    
    def scrape_uspto_patents(self, business_names: List[str], force_refresh: bool = False) -> Dict[str, int]:
        """Scrape patent data for businesses with a force refresh option."""
        patent_counts = {}
        
        if self.config.SKIP_PATENT_SEARCH:
            logger.info("Patent search is disabled via SKIP_PATENT_SEARCH configuration.")
            return {name: 0 for name in business_names}
        
        cache_file = "patent_cache.json"
        cached_patents = {}
        if os.path.exists(cache_file) and not force_refresh:
            try:
                with open(cache_file, 'r') as f:
                    cached_patents = json.load(f)
                logger.info(f"Loaded {len(cached_patents)} cached patent records.")
            except Exception as e:
                logger.warning(f"Could not load patent cache: {e}")
        else:
            logger.info("Bypassing patent cache due to force_refresh=True.")
    
        uncached_businesses = []
        if not force_refresh:
            for name in business_names:
                clean_name = self._clean_business_name(name).upper()
                if clean_name in cached_patents:
                    patent_counts[name] = cached_patents[clean_name]
                else:
                    uncached_businesses.append(name)
        else:
            uncached_businesses = business_names
    
        logger.info(f"Found {len(patent_counts)} cached results, need to search {len(uncached_businesses)} businesses.")
        
        if uncached_businesses:
            try:
                if self.config.USPTO_API_KEY:
                    logger.info("Using batch USPTO API search for patent data.")
                    from .patent_batch_optimizer import BatchPatentSearcher
                    
                    batch_searcher = BatchPatentSearcher(self.config)
                    # Also pass major institutions to the searcher
                    major_institutions = [
                        "University of Kansas", "University of Missouri-Kansas City", "Kansas State University",
                        "Children's Mercy Hospital", "St. Luke's Health System"
                    ]
                    combined_search_list = list(set(uncached_businesses + major_institutions))
                    new_patents = batch_searcher.batch_search_patents(combined_search_list)
                    
                    stats = batch_searcher.get_patent_statistics(new_patents)
                    logger.info(f"Batch search stats: {stats['businesses_with_patents']}/{stats['total_businesses']} entities have patents. Total patents: {stats['total_patents']}")
                else:
                    logger.warning("No USPTO API key found, using simulated data.")
                    new_patents = self._get_simulated_patent_data(uncached_businesses)
                
                patent_counts.update(new_patents)
                
                for name, count in new_patents.items():
                    clean_name = self._clean_business_name(name).upper()
                    cached_patents[clean_name] = count
                
                with open(cache_file, 'w') as f:
                    json.dump(cached_patents, f)
                logger.info(f"Updated patent cache with {len(new_patents)} new records")
                    
            except Exception as e:
                logger.error(f"Error in batch patent search: {e}", exc_info=True)
                logger.info("Falling back to individual patent searches.")
                new_patents = self._fetch_real_uspto_data(uncached_businesses)
                patent_counts.update(new_patents)
            
        return patent_counts
    
    def _get_from_cache(self, key: str) -> any:
        """Helper to get data from cache."""
        try:
            from utils.cache import cache
            if cache and cache.app:
                return cache.get(key)
            return None
        except (ImportError, RuntimeError):
            return None
    
    def _set_to_cache(self, key: str, value: any, timeout: int):
        """Helper to set data to cache."""
        try:
            from utils.cache import cache
            if cache and cache.app:
                cache.set(key, value, timeout=timeout)
        except (ImportError, RuntimeError):
            pass
    
    def _fetch_real_uspto_data(self, business_names: List[str]) -> Dict[str, int]:
        """Fetch real patent data from USPTO PatentsView API"""
        patent_counts = {}
        
        # PatentsView API endpoint
        search_url = f"{self.config.USPTO_API_URL}patent"
        
        headers = {
            'Accept': 'application/json',
            'X-Api-Key': self.config.USPTO_API_KEY
        }
        
        # First, let's try a location-based search for Kansas City area patents
        self._fetch_kc_area_patents(headers)
        
        # Process businesses in batches to avoid overwhelming the API
        batch_size = 5  # Smaller batch size for PatentsView API (45 req/min limit)
        
        # Keep track of searched companies to avoid duplicate searches
        searched_companies = set()
        failed_searches = set()  # Track failed searches to avoid retrying
        max_retries_per_company = 2  # Maximum retries for each company
        company_retry_count = {}  # Track retries per company
        
        # Log progress
        total_businesses = len(business_names)
        processed = 0
        
        for i in range(0, len(business_names), batch_size):
            batch = business_names[i:i + batch_size]
            
            for business_name in batch:
                try:
                    # Clean business name for search
                    clean_name = self._clean_business_name(business_name)
                    clean_name_upper = clean_name.upper()
                    
                    # Skip if we've already searched for this company
                    if clean_name_upper in searched_companies:
                        logger.debug(f"Skipping duplicate search for {clean_name}")
                        patent_counts[business_name] = 0
                        continue
                    
                    # Skip if this company has failed too many times
                    if clean_name_upper in failed_searches:
                        logger.debug(f"Skipping failed search for {clean_name}")
                        patent_counts[business_name] = 0
                        continue
                    
                    # Check retry count
                    retry_count = company_retry_count.get(clean_name_upper, 0)
                    if retry_count >= max_retries_per_company:
                        logger.warning(f"Max retries reached for {clean_name}, marking as failed")
                        failed_searches.add(clean_name_upper)
                        patent_counts[business_name] = 0
                        continue
                    
                    searched_companies.add(clean_name_upper)
                    processed += 1
                    
                    # Log progress every 100 businesses
                    if processed % 100 == 0:
                        logger.info(f"Patent search progress: {processed}/{total_businesses} businesses processed")
                    
                    # Build query for assignee organization
                    # Try multiple search strategies for better matching
                    
                    # First try exact match
                    query = {
                        "assignees.assignee_organization": clean_name
                    }
                    
                    # If company name is simple, also try with common suffixes
                    if len(clean_name.split()) == 1:
                        # For single word companies, try variations
                        variations = [
                            clean_name,
                            f"{clean_name} Corporation",
                            f"{clean_name} Inc",
                            f"{clean_name} LLC",
                            f"{clean_name} COMMUNICATIONS",  # For telecom companies
                            f"{clean_name} INNOVATION"  # For tech companies
                        ]
                        
                        query = {
                            "_or": [{"assignees.assignee_organization": var} for var in variations]
                        }
                    
                    # Request parameters
                    params = {
                        'q': json.dumps(query),
                        'f': json.dumps(["patent_id"]),
                        'o': json.dumps({"size": 1})  # Just get count, not all records
                    }
                    
                    logger.debug(f"Searching patents for: {clean_name}")
                    response = requests.get(search_url, headers=headers, params=params, timeout=self.config.API_TIMEOUT_LONG)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Extract patent count from response
                        if 'total_hits' in data:
                            patent_count = data['total_hits']
                            patent_counts[business_name] = patent_count
                            logger.info(f"Found {patent_count} patents for {clean_name}")
                        else:
                            patent_counts[business_name] = 0
                    elif response.status_code == 404:
                        # No patents found
                        patent_counts[business_name] = 0
                    else:
                        logger.warning(f"USPTO API error for {clean_name}: {response.status_code}")
                        if response.text:
                            logger.error(f"Response: {response.text[:500]}")
                            logger.error(f"Query was: {json.dumps(query, indent=2)}")
                            logger.error(f"Full URL: {response.url}")
                        patent_counts[business_name] = 0
                        # Track retry count
                        company_retry_count[clean_name_upper] = retry_count + 1
                        
                    # Rate limiting - PatentsView allows 45 requests per minute
                    time.sleep(1.5)  # 1.5 seconds between requests to stay under limit
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout searching patents for {business_name}")
                    patent_counts[business_name] = 0
                    company_retry_count[clean_name_upper] = retry_count + 1
                except Exception as e:
                    logger.error(f"Error fetching patents for {business_name}: {e}")
                    patent_counts[business_name] = 0
                    company_retry_count[clean_name_upper] = retry_count + 1
        
        # Only search for KC-based companies if we haven't found many patents
        if sum(patent_counts.values()) < 10:
            logger.info("Searching for additional KC-area patents...")
            # Only search companies that weren't already found
            remaining_companies = [name for name in business_names if patent_counts.get(name, 0) == 0][:100]
            if remaining_companies:
                kc_patents = self._fetch_kc_company_patents(headers, remaining_companies)
                for company, count in kc_patents.items():
                    if company in patent_counts:
                        patent_counts[company] = max(patent_counts[company], count)
        
        # Log final statistics
        total_patents = sum(patent_counts.values())
        companies_with_patents = sum(1 for count in patent_counts.values() if count > 0)
        logger.info(f"Patent search complete: {len(patent_counts)} businesses searched, {companies_with_patents} have patents, {total_patents} total patents found")
        logger.info(f"Failed searches: {len(failed_searches)}")
        
        return patent_counts
    
    def _clean_business_name(self, name: str) -> str:
        """Clean business name for USPTO search"""
        # Remove common suffixes
        suffixes = ['Inc', 'LLC', 'Corp', 'Corporation', 'Company', 'Co', 
                   'Ltd', 'Limited', 'Partners', 'LP', 'LLP', 'Solutions']
        
        clean_name = name
        for suffix in suffixes:
            # Remove with various punctuation
            clean_name = re.sub(rf'\s*[,.]?\s*{suffix}\.?\s*$', '', clean_name, flags=re.IGNORECASE)
            clean_name = re.sub(rf'\s+{suffix}\s+#\d+$', '', clean_name, flags=re.IGNORECASE)
        
        # Remove trailing numbers (like #1000)
        clean_name = re.sub(r'\s*#\d+$', '', clean_name)
        
        # Remove extra whitespace
        clean_name = ' '.join(clean_name.split())
        
        return clean_name.strip()
    
    def _fetch_kc_area_patents(self, headers: Dict) -> None:
        """Fetch general patent statistics for Kansas City area"""
        try:
            search_url = f"{self.config.USPTO_API_URL}patent"
            
            # Query for KC area patents using correct nested field format
            kc_query = {
                "_and": [
                    {
                        "_or": [
                            {"assignees.assignee_city": "Kansas City"},
                            {"assignees.assignee_city": "Overland Park"},
                            {"assignees.assignee_city": "Olathe"},
                            {"assignees.assignee_city": "Lenexa"}
                        ]
                    },
                    {
                        "_or": [
                            {"assignees.assignee_state": "MO"},
                            {"assignees.assignee_state": "KS"}
                        ]
                    }
                ]
            }
            
            params = {
                'q': json.dumps(kc_query),
                'f': json.dumps(["patent_id", "patent_title", "assignees"]),
                'o': json.dumps({"size": 10})  # Get top 10 to see sample data
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=self.config.API_TIMEOUT_LONG)
            
            if response.status_code == 200:
                data = response.json()
                total_kc_patents = data.get('total_hits', 0)
                logger.info(f"Found {total_kc_patents} total patents in Kansas City area")
                
                # Log sample patent holders
                if 'patents' in data and data['patents']:
                    logger.info("Sample KC area patent holders:")
                    for patent in data['patents'][:5]:
                        if 'assignees' in patent and patent['assignees']:
                            for assignee in patent['assignees']:
                                org = assignee.get('assignee_organization', 'Unknown')
                                city = assignee.get('assignee_city', 'Unknown')
                                if city in ['Kansas City', 'Overland Park', 'Olathe', 'Lenexa']:
                                    logger.info(f"  - {org} ({city})")
            else:
                logger.warning(f"KC area patent search failed: {response.status_code}")
                if response.text:
                    logger.error(f"KC search response: {response.text[:500]}")
                    logger.error(f"KC query was: {json.dumps(kc_query, indent=2)}")
                    logger.error(f"Full URL: {response.url}")
                
        except Exception as e:
            logger.error(f"Error fetching KC area patents: {e}")
    
    def _fetch_kc_company_patents(self, headers: Dict, business_names: List[str]) -> Dict[str, int]:
        """Search for patents from KC area that match company names"""
        patent_counts = {}
        search_url = f"{self.config.USPTO_API_URL}patent"
        
        # Get KC area patents with full assignee info
        kc_query = {
            "_and": [
                {
                    "_or": [
                        {"assignees.assignee_city": "Kansas City"},
                        {"assignees.assignee_city": "Overland Park"},
                        {"assignees.assignee_city": "Olathe"},
                        {"assignees.assignee_city": "Lenexa"},
                        {"assignees.assignee_city": "Independence"}
                    ]
                },
                {
                    "_or": [
                        {"assignees.assignee_state": "MO"},
                        {"assignees.assignee_state": "KS"}
                    ]
                }
            ]
        }
        
        params = {
            'q': json.dumps(kc_query),
            'f': json.dumps(["assignees"]),
            'o': json.dumps({"size": 1000})  # Get more results to match companies
        }
        
        try:
            response = requests.get(search_url, headers=headers, params=params, timeout=self.config.API_TIMEOUT_LONG)
            if response.status_code == 200:
                data = response.json()
                
                # Count patents by organization
                org_patent_counts = {}
                if 'patents' in data and data['patents']:
                    for patent in data['patents']:
                        if 'assignees' in patent and patent['assignees']:
                            for assignee in patent['assignees']:
                                org = assignee.get('assignee_organization', '')
                                if org:
                                    org_patent_counts[org] = org_patent_counts.get(org, 0) + 1
                
                # Match against our business names
                for business_name in business_names:
                    clean_name = self._clean_business_name(business_name).upper()
                    
                    # Look for matches in org names
                    for org, count in org_patent_counts.items():
                        org_upper = org.upper()
                        # Check if the business name is contained in the org name
                        if clean_name in org_upper or org_upper.startswith(clean_name):
                            if business_name not in patent_counts:
                                patent_counts[business_name] = count
                                logger.info(f"Found {count} patents for {business_name} as '{org}'")
                
        except Exception as e:
            logger.error(f"Error in KC company patent search: {e}")
        
        return patent_counts
    
    def _get_simulated_patent_data(self, business_names: List[str]) -> Dict[str, int]:
        """Get simulated patent data as fallback"""
        patent_counts = {}
        
        # Simulate patent data based on business type
        for name in business_names:
            if any(term in name.lower() for term in ["tech", "digital", "software", "systems"]):
                patent_counts[name] = random.randint(2, 8)
            elif any(term in name.lower() for term in ["bio", "pharma", "medical", "research"]):
                patent_counts[name] = random.randint(3, 12)
            elif any(term in name.lower() for term in ["manufacturing", "precision", "advanced"]):
                patent_counts[name] = random.randint(1, 4)
            else:
                patent_counts[name] = random.randint(0, 2)
                
        return patent_counts
    
    def scrape_infrastructure_data(self) -> List[Dict]:
        """Scrape infrastructure asset data"""
        infrastructure = []
        
        # Real KC infrastructure data
        infrastructure.extend([
            # Rail infrastructure (KC is 2nd largest rail center)
            {
                "type": "rail",
                "name": "BNSF Railway - Argentine Yard",
                "capacity": 1200000,  # tons/year
                "county": "Wyandotte County, KS",
                "description": "Major classification yard"
            },
            {
                "type": "rail",
                "name": "Union Pacific - Neff Yard",
                "capacity": 1100000,
                "county": "Jackson County, MO",
                "description": "Intermodal facility"
            },
            {
                "type": "rail",
                "name": "Norfolk Southern - Voltz Yard",
                "capacity": 900000,
                "county": "Jackson County, MO",
                "description": "Regional hub"
            },
            {
                "type": "rail",
                "name": "CPKC - Mill Street Yard",
                "capacity": 950000,
                "county": "Jackson County, MO",
                "description": "International gateway"
            },
            
            # Highway infrastructure
            {
                "type": "highway",
                "name": "Interstate 35",
                "capacity": 165000,  # vehicles/day
                "county": "Johnson County, KS",
                "description": "NAFTA Superhighway"
            },
            {
                "type": "highway", 
                "name": "Interstate 70",
                "capacity": 155000,
                "county": "Jackson County, MO",
                "description": "Major east-west corridor"
            },
            {
                "type": "highway",
                "name": "Interstate 435",
                "capacity": 140000,
                "county": "Johnson County, KS",
                "description": "Beltway system"
            },
            {
                "type": "highway",
                "name": "Interstate 29",
                "capacity": 120000,
                "county": "Platte County, MO",
                "description": "North-south corridor"
            },
            
            # Logistics facilities
            {
                "type": "logistics_park",
                "name": "Logistics Park Kansas City",
                "capacity": 17000000,  # square feet
                "county": "Jackson County, MO",
                "description": "Master-planned distribution hub"
            },
            {
                "type": "logistics_park",
                "name": "CenterPoint Intermodal Center",
                "capacity": 8000000,
                "county": "Johnson County, KS",
                "description": "Rail-served logistics campus"
            },
            {
                "type": "logistics_park",
                "name": "NorthPoint Development Parks",
                "capacity": 12000000,
                "county": "Clay County, MO",
                "description": "Multiple distribution centers"
            },
            
            # Airports
            {
                "type": "airport",
                "name": "Kansas City International Airport",
                "capacity": 10500000,  # passengers/year
                "county": "Platte County, MO",
                "description": "New terminal opened 2023"
            },
            
            # Ports
            {
                "type": "port",
                "name": "Port KC",
                "capacity": 800000,  # tons/year
                "county": "Jackson County, MO",
                "description": "Missouri River port"
            },
            
            # Utilities
            {
                "type": "utility",
                "name": "Evergy Power Grid",
                "capacity": 6500,  # MW
                "county": "Jackson County, MO",
                "description": "Regional power provider"
            },
            {
                "type": "utility",
                "name": "Google Fiber Network",
                "capacity": 1000,  # Gbps
                "county": "Jackson County, MO",
                "description": "High-speed internet"
            }
        ])
        
        return infrastructure
    
    def estimate_business_revenue(self, business: Dict) -> float:
        """Estimate business revenue based on industry benchmarks"""
        # Revenue per employee benchmarks by NAICS code (2-digit prefix)
        # Based on US Census Bureau Statistics of U.S. Businesses data
        revenue_per_employee = {
            # Transportation and Warehousing (48-49)
            "48": 250000,  # Transportation
            "49": 180000,  # Warehousing and Storage
            
            # Information (51)
            "51": 450000,  # Software/IT typically higher revenue per employee
            
            # Professional, Scientific, and Technical Services (54)
            "54": 200000,  # Professional services
            
            # Manufacturing (31-33)
            "31": 350000,  # Food Manufacturing
            "32": 400000,  # Chemical/Pharma Manufacturing
            "33": 380000,  # Machinery/Equipment Manufacturing
            
            # Healthcare and Social Assistance (62)
            "62": 150000,  # Healthcare services
            
            # Agriculture (11)
            "11": 180000,  # Agriculture/Animal Production
            
            # Wholesale Trade (42)
            "42": 600000,  # Wholesale typically high revenue/low employee
            
            # Retail Trade (44-45)
            "44": 200000,  # Retail Trade
            "45": 220000,  # Specialty Retail
            
            # Finance and Insurance (52)
            "52": 500000,  # Finance typically high revenue per employee
        }
        
        # Get employee count
        employees = business.get('employees', 0)
        if employees <= 0:
            # Estimate based on business type if no employee data
            employees = self._estimate_employee_count(business)
        
        # Get NAICS code
        naics = str(business.get('naics_code', ''))
        if len(naics) >= 2:
            naics_2digit = naics[:2]
            base_revenue_per_emp = revenue_per_employee.get(naics_2digit, 250000)  # Default
        else:
            # Try to infer from business name/type
            base_revenue_per_emp = self._infer_revenue_benchmark(business.get('name', ''))
        
        # Adjust for business age (newer businesses typically have lower revenue)
        years_in_business = business.get('years_in_business', 5)
        age_factor = min(1.0, (years_in_business + 2) / 7)  # Ramps up over 5 years
        
        # Adjust for innovation indicators
        innovation_boost = 1.0
        if business.get('patent_count', 0) > 0:
            innovation_boost += 0.1
        if business.get('sbir_awards', 0) > 0:
            innovation_boost += 0.15
        
        # Calculate estimated revenue
        estimated_revenue = employees * base_revenue_per_emp * age_factor * innovation_boost
        
        # Apply reasonable bounds
        # Small business: $50K - $10M
        # Medium business: $10M - $100M  
        # Large business: $100M+
        if employees < 20:
            estimated_revenue = max(50000, min(10000000, estimated_revenue))
        elif employees < 500:
            estimated_revenue = max(1000000, min(100000000, estimated_revenue))
        
        return estimated_revenue
    
    def _estimate_employee_count(self, business: Dict) -> int:
        """Estimate employee count based on business type"""
        name = business.get('name', '').lower()
        
        # Keywords indicating business size
        if any(term in name for term in ['corporation', 'corp', 'international', 'global']):
            return 100
        elif any(term in name for term in ['inc', 'company', 'co', 'group']):
            return 25
        elif any(term in name for term in ['llc', 'associates', 'partners']):
            return 10
        elif any(term in name for term in ['consulting', 'services', 'solutions']):
            return 8
        else:
            return 5  # Default small business
    
    def _infer_revenue_benchmark(self, business_name: str) -> float:
        """Infer revenue per employee from business name"""
        name_lower = business_name.lower()
        
        # Technology/Software
        if any(term in name_lower for term in ['software', 'tech', 'digital', 'cyber', 'data']):
            return 400000
        # Logistics/Transportation
        elif any(term in name_lower for term in ['logistics', 'transport', 'freight', 'shipping']):
            return 250000
        # Manufacturing
        elif any(term in name_lower for term in ['manufacturing', 'industrial', 'production']):
            return 350000
        # Healthcare/Bio
        elif any(term in name_lower for term in ['health', 'medical', 'bio', 'pharma']):
            return 300000
        # Professional Services
        elif any(term in name_lower for term in ['consulting', 'advisory', 'legal', 'accounting']):
            return 200000
        else:
            return 250000  # Default
    
    def scrape_market_data(self) -> Dict:
        """Scrape market trends and commodity prices"""
        # Base market data with growth rates
        market_data = {
            "logistics_growth": {
                "national": 0.063,  # 6.3% CAGR
                "regional": 0.058,  # 5.8% KC specific
                "ecommerce_impact": 1.15  # 15% boost
            },
            "biotech_growth": {
                "national": 0.072,  # 7.2% CAGR
                "regional": 0.065,
                "sbir_funding_trend": 1.08  # 8% increase
            },
            "manufacturing_growth": {
                "national": 0.041,  # 4.1% CAGR
                "regional": 0.038,
                "automation_factor": 0.95  # 5% job reduction
            },
            "tech_growth": {
                "national": 0.089,  # 8.9% CAGR
                "regional": 0.082,
                "startup_activity": 1.12  # 12% increase
            },
            "commodity_prices": {
                "diesel_fuel": 3.85,  # $/gallon
                "warehouse_rent": 7.25,  # $/sq ft/year
                "industrial_electricity": 0.068,  # $/kWh
                "natural_gas": 2.54  # $/thousand cubic feet
            }
        }
        
        # Enhance with real-time market monitoring if enabled
        use_real_time_monitoring = os.getenv('USE_REAL_TIME_MARKET_DATA', 'false').lower() == 'true'
        
        if use_real_time_monitoring:
            try:
                from .market_monitor import MarketMonitor
                monitor = MarketMonitor()
                market_data = monitor.enhance_market_data(market_data)
                logger.info("Enhanced market data with real-time indicators")
                
                # Also enhance with geopolitical risk data
                from .geopolitical_analyzer import GeopoliticalAnalyzer
                geo_analyzer = GeopoliticalAnalyzer()
                market_data = geo_analyzer.enhance_with_geopolitical_data(market_data)
                logger.info("Enhanced market data with geopolitical risk analysis")
            except Exception as e:
                logger.warning(f"Could not enhance market data: {e}")
        
        return market_data
    
    def scrape_university_data(self) -> List[Dict]:
        """Scrape university research and partnership data with NSF grants"""
        universities = []
        
        # Start with existing static data
        uni_data = [
            {
                "name": "University of Missouri-Kansas City",
                "research_expenditure": 45000000,
                "stem_graduates": 1200,
                "business_partnerships": 35,
                "focus_areas": ["Biosciences", "Engineering", "Computer Science"],
                "nsf_grants": []
            },
            {
                "name": "University of Kansas",
                "research_expenditure": 275000000,
                "stem_graduates": 3500,
                "business_partnerships": 85,
                "focus_areas": ["Pharmaceuticals", "Engineering", "Biosciences"],
                "nsf_grants": []
            },
            {
                "name": "Kansas State University",
                "research_expenditure": 188000000,
                "stem_graduates": 2800,
                "business_partnerships": 65,
                "focus_areas": ["Agriculture", "Engineering", "Animal Health"],
                "nsf_grants": []
            },
            {
                "name": "Johnson County Community College",
                "research_expenditure": 5000000,
                "stem_graduates": 800,
                "business_partnerships": 25,
                "focus_areas": ["IT", "Healthcare", "Business"],
                "nsf_grants": []
            }
        ]
        
        # Fetch NSF grants for each university
        for uni in uni_data:
            uni["nsf_grants"] = self._fetch_nsf_grants_for_university(uni["name"])
            
            # Update research expenditure with actual NSF funding
            nsf_total = sum(grant.get("amount", 0) for grant in uni["nsf_grants"])
            if nsf_total > 0:
                # NSF grants represent a portion of total research funding
                uni["nsf_funding"] = nsf_total
                uni["research_clusters"] = self._categorize_research_clusters(uni["nsf_grants"])
        
        return uni_data
    
    def _fetch_nsf_grants_for_university(self, university_name: str) -> List[Dict]:
        """Fetch NSF grants for a specific university"""
        logger.info(f"Fetching NSF grants for {university_name}")
        
        base_url = "https://www.research.gov/awardapi-service/v1/awards.json"
        grants = []
        
        try:
            # Map university names to NSF search terms and expected awardee names
            search_mapping = {
                "University of Missouri-Kansas City": {
                    "search_terms": ["Kansas City", "UMKC"],
                    "awardee_patterns": ["CURATORS OF THE UNIVERSITY OF MISSOURI", "UNIVERSITY OF MISSOURI"]
                },
                "University of Kansas": {
                    "search_terms": ["Kansas City", "Lawrence Kansas"],
                    "awardee_patterns": ["UNIVERSITY OF KANSAS CENTER FOR RESEARCH", "UNIVERSITY OF KANSAS"]
                },
                "Kansas State University": {
                    "search_terms": ["Kansas State", "Manhattan Kansas"],
                    "awardee_patterns": ["KANSAS STATE UNIVERSITY"]
                },
                "Johnson County Community College": {
                    "search_terms": ["Johnson County", "JCCC"],
                    "awardee_patterns": ["JOHNSON COUNTY COMMUNITY COLLEGE", "JCCC"]
                }
            }
            
            mapping = search_mapping.get(university_name, {
                "search_terms": [university_name],
                "awardee_patterns": [university_name.upper()]
            })
            
            search_terms = mapping["search_terms"]
            awardee_patterns = mapping["awardee_patterns"]
            
            for term in search_terms:
                params = {
                    "awardeeName": term,
                    "printFields": "id,title,fundsObligatedAmt,startDate,abstractText,piFirstName,piLastName,awardee"
                }
                
                response = requests.get(base_url, params=params, timeout=self.config.API_TIMEOUT_MEDIUM)
                logger.debug(f"NSF API URL: {response.url}")
                if response.status_code == 200:
                    data = response.json()
                    awards = data.get("response", {}).get("award", [])
                    logger.debug(f"NSF API returned {len(awards)} awards for term '{term}'")
                    
                    for award in awards:
                        # Check if this award belongs to the university we're looking for
                        awardee = award.get("awardee", "").upper()
                        matches_pattern = any(pattern in awardee for pattern in awardee_patterns)
                        
                        if matches_pattern:
                            # Only include recent grants (last 5 years)
                            start_date = award.get("startDate", "")
                            if start_date:
                                try:
                                    # NSF date format is MM/DD/YYYY
                                    if "/" in start_date:
                                        parts = start_date.split("/")
                                        if len(parts) >= 3:
                                            year = int(parts[2])
                                    else:
                                        # Fallback for YYYY-MM-DD format
                                        year = int(start_date[:4])
                                    
                                    if year >= 2019:
                                        grants.append({
                                            "id": award.get("id"),
                                            "title": award.get("title", ""),
                                            "amount": int(award.get("fundsObligatedAmt", 0)),
                                            "start_date": start_date,
                                            "abstract": award.get("abstractText", "")[:500],  # First 500 chars
                                            "pi": f"{award.get('piFirstName', '')} {award.get('piLastName', '')}",
                                            "institution": award.get("awardee", "")
                                        })
                                except ValueError:
                                    # Skip if year can't be parsed
                                    pass
            
            logger.info(f"Found {len(grants)} NSF grants for {university_name}")
            
        except Exception as e:
            logger.error(f"Error fetching NSF grants for {university_name}: {e}")
        
        return grants
    
    def _categorize_research_clusters(self, grants: List[Dict]) -> Dict[str, int]:
        """Categorize NSF grants by research cluster type"""
        cluster_keywords = {
            "biosciences": ["biomedical", "biology", "cancer", "drug", "pharmaceutical", "genetics", 
                          "medical", "health", "disease", "therapy", "clinical"],
            "technology": ["artificial intelligence", "machine learning", "computer", "software", 
                         "data", "algorithm", "cyber", "information", "digital", "AI", "ML"],
            "manufacturing": ["manufacturing", "industrial", "production", "materials", "engineering",
                            "mechanical", "process", "automation", "robotics"],
            "animal_health": ["animal", "veterinary", "agriculture", "livestock", "food", "crop"],
            "logistics": ["supply chain", "logistics", "transportation", "distribution"]
        }
        
        cluster_counts = {cluster: 0 for cluster in cluster_keywords}
        cluster_amounts = {cluster: 0 for cluster in cluster_keywords}
        
        for grant in grants:
            # Check title and abstract for keywords
            text = (grant.get("title", "") + " " + grant.get("abstract", "")).lower()
            
            for cluster, keywords in cluster_keywords.items():
                if any(keyword in text for keyword in keywords):
                    cluster_counts[cluster] += 1
                    cluster_amounts[cluster] += grant.get("amount", 0)
        
        # Return both counts and total funding by cluster
        return {
            "counts": cluster_counts,
            "funding": cluster_amounts
        }
    
    def fetch_fred_data(self, series_id: str = "KSMSA28140URN") -> Optional[Dict]:
        """
        Fetch macroeconomic data from FRED (Federal Reserve Economic Data)
        Default series: Kansas City MSA unemployment rate
        """
        try:
            # FRED API endpoint
            base_url = "https://api.stlouisfed.org/fred/series/observations"
            
            # You would need an API key in production
            params = {
                'series_id': series_id,
                'api_key': self.config.FRED_API_KEY if hasattr(self.config, 'FRED_API_KEY') else 'demo',
                'file_type': 'json',
                'limit': 100,
                'sort_order': 'desc'
            }
            
            # For now, return mock data structure
            logger.info(f"Would fetch FRED data for series: {series_id}")
            return {
                'unemployment_rate': 3.8,
                'gdp_growth': 2.3,
                'inflation_rate': 2.5,
                'interest_rate': 5.25
            }
        except Exception as e:
            logger.error(f"Error fetching FRED data: {e}")
            return None
    
    def fetch_eia_data(self, area: str = "Kansas") -> Optional[Dict]:
        """
        Fetch energy price data from EIA (Energy Information Administration)
        """
        try:
            # EIA API endpoint
            base_url = "https://api.eia.gov/v2/electricity/retail-sales/data"
            
            # Mock data structure for energy costs
            logger.info(f"Would fetch EIA energy data for: {area}")
            return {
                'commercial_electricity_rate': 10.5,  # cents per kWh
                'industrial_electricity_rate': 7.8,
                'natural_gas_price': 8.45,  # $ per thousand cubic feet
                'energy_cost_index': 95  # relative to national average (100)
            }
        except Exception as e:
            logger.error(f"Error fetching EIA data: {e}")
            return None
    
    def fetch_alpha_vantage_data(self, symbol: str = "SPY") -> Optional[Dict]:
        """
        Fetch market data from Alpha Vantage
        """
        try:
            # Alpha Vantage API endpoint
            base_url = "https://www.alphavantage.co/query"
            
            # Mock market sentiment data
            logger.info(f"Would fetch Alpha Vantage market data for: {symbol}")
            return {
                'market_sentiment': 'neutral',
                'volatility_index': 18.5,
                'sector_performance': {
                    'technology': 0.08,
                    'healthcare': 0.06,
                    'industrials': 0.04,
                    'financials': 0.03
                }
            }
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data: {e}")
            return None
    
    def integrate_federal_data_sources(self, businesses: List[Dict]) -> List[Dict]:
        """
        Integrate multiple federal data sources with weighted contributions
        Based on Akinyede & Caruso (2025) methodology:
        - USPTO: 20% weight
        - SBIR: 15% weight  
        - BLS: 20% weight
        - EIA: 15% weight
        - FRED: 15% weight
        - Alpha Vantage: 15% weight
        """
        logger.info("Integrating federal data sources with weighted contributions...")
        
        # Fetch macro data once (not per business)
        fred_data = self.fetch_fred_data()
        eia_data = self.fetch_eia_data()
        market_data = self.fetch_alpha_vantage_data()
        
        # Apply data to businesses
        for business in businesses:
            # Add macro context
            if fred_data:
                business['macro_unemployment'] = fred_data.get('unemployment_rate')
                business['macro_gdp_growth'] = fred_data.get('gdp_growth')
            
            if eia_data:
                business['energy_cost_index'] = eia_data.get('energy_cost_index')
                
            if market_data:
                # Match business to sector performance
                naics = business.get('naics_code', '')
                if naics.startswith('54'):  # Tech services
                    business['sector_momentum'] = market_data['sector_performance'].get('technology', 0)
                elif naics.startswith('325'):  # Chemicals/Pharma
                    business['sector_momentum'] = market_data['sector_performance'].get('healthcare', 0)
                elif naics.startswith('33'):  # Manufacturing
                    business['sector_momentum'] = market_data['sector_performance'].get('industrials', 0)
                elif naics.startswith('52'):  # Finance
                    business['sector_momentum'] = market_data['sector_performance'].get('financials', 0)
        
        logger.info(f"Enhanced {len(businesses)} businesses with federal data")
        return businesses
    
    def run_full_scraping_cycle(self) -> Dict:
        """Run complete data collection cycle with intelligent caching
        
        This caches raw data that doesn't depend on user parameters.
        The analysis results are NOT cached here.
        """
        logger.info("Starting full scraping cycle")
        start_time = datetime.now()
        
        # Import cache utilities
        try:
            from utils.cache import cache, init_cache
            # Mock Flask app for standalone script
            class MockApp:
                config = {}
            mock_app = MockApp()
            init_cache(mock_app)
        except ImportError:
            logger.warning("Cache utilities not available, proceeding without caching")
            cache = None
        
        # Fast path: load preloaded final dataset if configured
        try:
            preloaded = self.config.FINAL_DATASET_CSV or os.getenv('FINAL_DATASET_CSV')
            if preloaded and os.path.exists(preloaded):
                logger.info(f"Using preloaded dataset from {preloaded}")
                df = pd.read_csv(preloaded)
                # Normalize types and map to expected business schema
                def to_str(x):
                    return str(x) if pd.notna(x) else ""
                def to_num(x, dtype=float):
                    try:
                        return dtype(x)
                    except Exception:
                        return None
                current_year = datetime.now().year
                businesses = []
                for _, r in df.iterrows():
                    yr = pd.to_numeric(r.get('year_established'), errors='coerce')
                    if pd.isna(yr):
                        yr = None
                    rec = {
                        'name': to_str(r.get('name')),
                        'address': to_str(r.get('address')),
                        'city': to_str(r.get('city')),
                        'state': to_str(r.get('state')),
                        'zip': to_str(r.get('zip')),
                        'county': to_str(r.get('county')),
                        'county_class': to_str(r.get('county_class')) if 'county_class' in r else None,
                        'in_focus_county': int(pd.to_numeric(r.get('in_focus_county'), errors='coerce')) if 'in_focus_county' in r and pd.notna(pd.to_numeric(r.get('in_focus_county'), errors='coerce')) else 0,
                        'naics_code': to_str(r.get('naics_code')),
                        'revenue_estimate': pd.to_numeric(r.get('revenue'), errors='coerce') if 'revenue' in r else None,
                        'employees': int(pd.to_numeric(r.get('employees'), errors='coerce')) if pd.notna(pd.to_numeric(r.get('employees'), errors='coerce')) else None,
                        'year_established': int(yr) if yr else None,
                        'lat': float(pd.to_numeric(r.get('lat'), errors='coerce')) if 'lat' in r else None,
                        'lon': float(pd.to_numeric(r.get('lon'), errors='coerce')) if 'lon' in r else None,
                        'cluster_type': to_str(r.get('cluster_type')) or None,
                        'data_source': 'preloaded_final_csv',
                        'years_in_business': (current_year - int(yr)) if yr else None,
                        'sbir_awards': 0,
                        'patent_count': 0,
                        'status': 'Active'
                    }
                    businesses.append(rec)

                # Build results skeleton and enrich with internal non-network data
                results = {
                    'businesses': businesses,
                    'sbir_awards': [],
                    'employment_data': self.scrape_bls_data(),
                    'infrastructure': self.scrape_infrastructure_data(),
                    'patent_counts': {},
                    'market_data': self.scrape_market_data(),
                    'university_data': self.scrape_university_data(),
                    'errors': []
                }
                logger.info(f"Loaded {len(businesses)} businesses from preloaded dataset")
                return results
        except Exception as e:
            logger.warning(f"Failed preloaded dataset path handling: {e}")

        # Check for cached raw data (valid for 1 hour)
        cache_key = "raw_business_data_kc_metro_v2"
        cached_data = None
        
        if cache:
            try:
                cached_data = cache.get(cache_key)
                if cached_data and isinstance(cached_data, dict):
                    # Validate cached data has expected structure
                    required_keys = ["businesses", "sbir_awards", "employment_data", 
                                   "infrastructure", "market_data", "university_data"]
                    if all(key in cached_data for key in required_keys):
                        logger.info(f"Using cached raw business data ({len(cached_data.get('businesses', []))} businesses)")
                        # Patent data is fetched separately as it has its own cache
                        return cached_data
                    else:
                        logger.warning("Cached data missing required keys, fetching fresh data")
            except Exception as e:
                logger.warning(f"Error retrieving cached data: {e}")
        
        results = {
            "businesses": [],
            "sbir_awards": [],
            "employment_data": {},
            "infrastructure": [],
            "patent_counts": {},
            "market_data": {},
            "university_data": [],
            "errors": []
        }
        
        try:
            # Scrape business registries
            logger.info("Scraping Kansas businesses...")
            kansas_businesses = self.scrape_kansas_businesses()
            results["businesses"].extend(kansas_businesses)
            
            logger.info("Scraping Missouri businesses...")
            missouri_businesses = self.scrape_missouri_businesses()
            results["businesses"].extend(missouri_businesses)
            
            # Scrape SBIR awards
            logger.info("Scraping SBIR/STTR awards...")
            results["sbir_awards"] = self.scrape_sbir_awards()
            
            # Match SBIR awards to businesses
            for business in results["businesses"]:
                business["sbir_awards"] = 0
                for award in results["sbir_awards"]:
                    if award["company"] in business["name"] or business["name"] in award["company"]:
                        business["sbir_awards"] += 1
            
            # Scrape employment data
            logger.info("Scraping BLS employment data...")
            results["employment_data"] = self.scrape_bls_data()
            
            # Scrape infrastructure
            logger.info("Scraping infrastructure data...")
            results["infrastructure"] = self.scrape_infrastructure_data()
            
            # Get patent counts for businesses
            if results["businesses"]:
                logger.info("Scraping patent data...")
                business_names = [b["name"] for b in results["businesses"]]
                
                # Log total businesses to search
                logger.info(f"Searching patents for {len(business_names)} businesses...")
                
                results["patent_counts"] = self.scrape_uspto_patents(business_names)
                
                # Add patent counts to businesses
                for business in results["businesses"]:
                    business["patent_count"] = results["patent_counts"].get(business["name"], 0)
            
            # Scrape market data
            logger.info("Scraping market trends...")
            results["market_data"] = self.scrape_market_data()
            
            # Scrape university data
            logger.info("Scraping university data...")
            results["university_data"] = self.scrape_university_data()
            
            # Log scraping results
            if self.session:
                log_entry = ScrapingLog(
                    data_source_id=1,
                    start_time=start_time,
                    end_time=datetime.now(),
                    records_scraped=len(results["businesses"]),
                    errors=results["errors"],
                    status="success" if not results["errors"] else "partial"
                )
                self.session.add(log_entry)
                self.session.commit()
                
        except Exception as e:
            logger.error(f"Error in scraping cycle: {e}")
            results["errors"].append(str(e))
            
        logger.info(f"Scraping cycle completed. Total businesses: {len(results['businesses'])}")
        
        # Cache the raw data for future use (1 hour TTL)
        if cache and not results["errors"]:
            try:
                # Don't cache patent_counts as they're cached separately
                cache_data = {
                    "businesses": results["businesses"],
                    "sbir_awards": results["sbir_awards"],
                    "employment_data": results["employment_data"],
                    "infrastructure": results["infrastructure"],
                    "market_data": results["market_data"],
                    "university_data": results["university_data"],
                    "patent_counts": results["patent_counts"],  # Include for completeness
                    "cached_at": datetime.now().isoformat()
                }
                cache.set(cache_key, cache_data, timeout=3600)  # 1 hour cache
                logger.info("Cached raw business data for future use")
            except Exception as e:
                logger.warning(f"Failed to cache raw data: {e}")
        
        return results
