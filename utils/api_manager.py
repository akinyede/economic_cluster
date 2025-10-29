"""Centralized API management with rate limiting and error handling"""
import time
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import requests
from collections import defaultdict
import json
from functools import wraps

logger = logging.getLogger(__name__)

class RateLimiter:
    """Token bucket rate limiter for API calls"""
    
    def __init__(self, rate: int, per: int = 60):
        """
        Args:
            rate: Number of requests allowed
            per: Time period in seconds (default 60 for per minute)
        """
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.updated_at = time.time()
        
    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, blocking if necessary.
        Returns wait time in seconds.
        """
        now = time.time()
        elapsed = now - self.updated_at
        
        # Refill tokens based on elapsed time
        self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / self.per))
        self.updated_at = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return 0.0
        else:
            # Calculate wait time
            deficit = tokens - self.tokens
            wait_time = deficit * self.per / self.rate
            return wait_time

class APIManager:
    """Manages API calls with rate limiting, retries, and caching"""
    
    def __init__(self):
        self.rate_limiters = {
            'uspto': RateLimiter(45, 60),  # 45 requests per minute
            'fred': RateLimiter(120, 60),   # 120 requests per minute
            'eia': RateLimiter(30, 60),     # 30 requests per minute (conservative)
            'alpha_vantage': RateLimiter(5, 60),  # 5 requests per minute (free tier)
        }
        self.request_cache = {}
        self.request_history = defaultdict(list)
        
    def rate_limited_request(self, api_name: str, url: str, 
                           params: Dict = None, 
                           timeout: int = 30,
                           max_retries: int = 3,
                           backoff_factor: float = 2.0,
                           cache_ttl: int = 3600) -> Optional[Dict]:
        """
        Make a rate-limited API request with retries and caching.
        
        Args:
            api_name: Name of the API for rate limiting
            url: API endpoint URL
            params: Query parameters
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            cache_ttl: Cache time-to-live in seconds
            
        Returns:
            Parsed JSON response or None on failure
        """
        # Check cache first
        cache_key = f"{api_name}:{url}:{json.dumps(params or {}, sort_keys=True)}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {api_name}")
            return cached
            
        # Rate limiting
        if api_name in self.rate_limiters:
            wait_time = self.rate_limiters[api_name].acquire()
            if wait_time > 0:
                logger.info(f"Rate limit reached for {api_name}, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        
        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(max_retries):
            try:
                # Adaptive timeout based on API
                if api_name == 'eia':
                    timeout = max(timeout, 60)  # EIA needs longer timeout
                    
                logger.debug(f"API request to {api_name}: attempt {attempt + 1}/{max_retries}")
                
                response = requests.get(url, params=params, timeout=timeout)
                
                # Record request for monitoring
                self._record_request(api_name, response.status_code)
                
                if response.status_code == 200:
                    data = response.json()
                    # Cache successful response
                    self._cache_response(cache_key, data, cache_ttl)
                    return data
                elif response.status_code == 429:
                    # Rate limit hit - wait longer
                    wait_time = (backoff_factor ** attempt) * 60
                    logger.warning(f"Rate limit 429 from {api_name}, waiting {wait_time}s")
                    time.sleep(wait_time)
                elif response.status_code >= 500:
                    # Server error - retry with backoff
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Server error {response.status_code} from {api_name}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    # Client error - don't retry
                    logger.error(f"Client error {response.status_code} from {api_name}: {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {timeout}s"
                wait_time = backoff_factor ** attempt
                logger.warning(f"Timeout on {api_name}, retrying in {wait_time}s")
                time.sleep(wait_time)
                
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {str(e)}"
                wait_time = backoff_factor ** attempt
                logger.warning(f"Connection error on {api_name}, retrying in {wait_time}s")
                time.sleep(wait_time)
                
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error calling {api_name}: {e}")
                # Don't retry on unexpected errors
                break
        
        logger.error(f"All retries exhausted for {api_name}. Last error: {last_error}")
        return None
    
    def _get_cached(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if not expired"""
        if cache_key in self.request_cache:
            cached_data, expiry = self.request_cache[cache_key]
            if datetime.now() < expiry:
                return cached_data
            else:
                del self.request_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, data: Dict, ttl: int):
        """Cache API response with TTL"""
        expiry = datetime.now() + timedelta(seconds=ttl)
        self.request_cache[cache_key] = (data, expiry)
    
    def _record_request(self, api_name: str, status_code: int):
        """Record request for monitoring"""
        self.request_history[api_name].append({
            'timestamp': datetime.now(),
            'status_code': status_code
        })
        
        # Keep only last hour of history
        cutoff = datetime.now() - timedelta(hours=1)
        self.request_history[api_name] = [
            r for r in self.request_history[api_name]
            if r['timestamp'] > cutoff
        ]
    
    def get_api_stats(self) -> Dict[str, Dict]:
        """Get statistics for API usage"""
        stats = {}
        for api_name, requests in self.request_history.items():
            if requests:
                success_count = sum(1 for r in requests if 200 <= r['status_code'] < 300)
                error_count = sum(1 for r in requests if r['status_code'] >= 400)
                stats[api_name] = {
                    'total_requests': len(requests),
                    'success_rate': success_count / len(requests) if requests else 0,
                    'error_count': error_count,
                    'requests_per_minute': len(requests) / 60  # Assuming history is last hour
                }
        return stats
    
    def batch_request(self, api_name: str, requests: list, 
                     batch_size: int = 10,
                     delay_between_batches: float = 1.0) -> list:
        """
        Process multiple API requests in batches to avoid overwhelming the API.
        
        Args:
            api_name: Name of the API
            requests: List of (url, params) tuples
            batch_size: Number of requests per batch
            delay_between_batches: Seconds to wait between batches
            
        Returns:
            List of responses (None for failed requests)
        """
        results = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(requests)-1)//batch_size + 1}")
            
            for url, params in batch:
                result = self.rate_limited_request(api_name, url, params)
                results.append(result)
            
            # Delay between batches
            if i + batch_size < len(requests):
                time.sleep(delay_between_batches)
                
        return results

# Global API manager instance
api_manager = APIManager()

def with_rate_limit(api_name: str):
    """Decorator for rate-limited API calls"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract URL and params from function
            url = kwargs.get('url') or (args[0] if args else None)
            params = kwargs.get('params', {})
            
            # Use API manager for the request
            result = api_manager.rate_limited_request(
                api_name, 
                url, 
                params,
                timeout=kwargs.get('timeout', 30),
                max_retries=kwargs.get('max_retries', 3)
            )
            
            # Update kwargs with the result
            kwargs['response_data'] = result
            return func(*args, **kwargs)
        return wrapper
    return decorator