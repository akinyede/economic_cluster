"""Caching configuration and utilities for KC Cluster Prediction Tool"""
import os
import json
import hashlib
from flask_caching import Cache as FlaskCache
from functools import wraps
from typing import Any, Dict, Optional
import logging
from flask import current_app, Flask

logger = logging.getLogger(__name__)

class Cache:
    """
    A wrapper for Flask-Caching to handle initialization and provide caching utilities.
    It can be initialized with a Flask app instance to avoid relying on `current_app`,
    making it safe to use from background threads if the instance is created with the app.
    """
    def __init__(self, app: Optional[Flask] = None):
        self._app = app
        self._cache = FlaskCache()
        if app:
            self.init_app(app)

    def init_app(self, app: Flask):
        """Initializes the cache with a Flask app instance."""
        self._app = app
        # Try Redis first, fallback to simple cache
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        # Check if Redis is explicitly disabled or if we should use simple cache
        force_simple = os.getenv('USE_SIMPLE_CACHE', 'false').lower() == 'true'
        
        if force_simple:
            # User explicitly wants simple cache
            cache_config = {
                'CACHE_TYPE': 'simple',
                'CACHE_DEFAULT_TIMEOUT': 3600,
                'CACHE_THRESHOLD': 500
            }
            app.config.update(cache_config)
            self._cache.init_app(app)
            logger.info("Cache initialized with SimpleCache (forced by config)")
            return
        
        try:
            # Try to connect to Redis with short timeout
            import redis
            r = redis.from_url(redis_url, socket_connect_timeout=0.5, socket_timeout=0.5)
            r.ping()
            
            cache_config = {
                'CACHE_TYPE': 'redis',
                'CACHE_REDIS_URL': redis_url,
                'CACHE_DEFAULT_TIMEOUT': 3600,  # 1 hour default
                'CACHE_KEY_PREFIX': 'kc_cluster_'
            }
            app.config.update(cache_config)
            self._cache.init_app(app)
            logger.info("Cache initialized with Redis at %s", redis_url)
        except ImportError:
            # Redis library not installed
            logger.info("Redis library not installed. Using SimpleCache for local development.")
            cache_config = {
                'CACHE_TYPE': 'simple',
                'CACHE_DEFAULT_TIMEOUT': 3600,
                'CACHE_THRESHOLD': 500
            }
            app.config.update(cache_config)
            self._cache.init_app(app)
            logger.info("Cache initialized with SimpleCache")
        except Exception as e:
            # Redis not running or connection failed
            # Only log as warning if explicitly configured to use Redis
            if 'REDIS_URL' in os.environ:
                logger.warning(f"Redis configured but connection failed: {e}. Falling back to SimpleCache.")
            else:
                logger.info(f"Redis not available (using SimpleCache). Install Redis for better performance: brew install redis")
            
            cache_config = {
                'CACHE_TYPE': 'simple',
                'CACHE_DEFAULT_TIMEOUT': 3600,
                'CACHE_THRESHOLD': 500
            }
            app.config.update(cache_config)
            self._cache.init_app(app)
            logger.info("Cache initialized with SimpleCache")

    @property
    def cache(self):
        """
        Returns the underlying cache instance, ensuring it's initialized.
        Uses the stored app if available, otherwise falls back to current_app.
        """
        if self._cache.app:
            return self._cache
        
        app = self._app or current_app
        if app:
            self.init_app(app)
            return self._cache

        raise RuntimeError("Flask-Caching has not been initialized with a Flask app.")

    def make_cache_key(self, *args, **kwargs):
        """Generate a cache key from function arguments"""
        key_parts = []
        for arg in args:
            if isinstance(arg, (dict, list, tuple)):
                key_parts.append(json.dumps(arg, sort_keys=True))
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (dict, list, tuple)):
                key_parts.append(f"{k}={json.dumps(v, sort_keys=True)}")
            else:
                key_parts.append(f"{k}={v}")
        
        key_string = "|".join(key_parts)
        return f"func_{hashlib.md5(key_string.encode()).hexdigest()}"

    def cache_result(self, timeout: int = 3600):
        """Decorator to cache function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = self.make_cache_key(func.__name__, *args, **kwargs)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                logger.debug(f"Cache miss for {func.__name__}")
                result = func(*args, **kwargs)
                self.cache.set(cache_key, result, timeout=timeout)
                return result
            return wrapper
        return decorator

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        return self.cache.get(key)

    def set(self, key: str, value: Any, timeout: int = 3600):
        """Set a value in the cache."""
        self.cache.set(key, value, timeout=timeout)
        logger.debug(f"Set cache key: {key}")

    def clear(self):
        """Clears the entire cache."""
        self.cache.clear()
        logger.info("Cache cleared.")
    
    @property
    def app(self):
        """Returns the Flask app instance if available."""
        return self._app or (self._cache.app if hasattr(self._cache, 'app') else None)

# A default instance to be initialized by the app factory
cache = Cache()