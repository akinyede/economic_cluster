"""Circuit breaker implementation for external API calls"""
import time
import logging
from typing import Optional, Callable, Any, Dict
from functools import wraps
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        name: str = "CircuitBreaker",
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.success_count = 0
        self._lock = Lock()

        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rejected_calls': 0,
            'state_changes': []
        }

    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            self.stats['total_calls'] += 1
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self.stats['rejected_calls'] += 1
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        return bool(self.last_failure_time and (time.time() - self.last_failure_time >= self.recovery_timeout))

    def _on_success(self):
        with self._lock:
            self.stats['successful_calls'] += 1
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= 2:
                    self._transition_to(CircuitState.CLOSED)

    def _on_failure(self):
        with self._lock:
            self.stats['failed_calls'] += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.success_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self.failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState):
        old = self.state
        self.state = new_state
        self.stats['state_changes'].append({'from': old.value, 'to': new_state.value, 'timestamp': time.time()})
        logger.info(f"Circuit breaker {self.name}: {old.value} -> {new_state.value}")
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0

    def get_state(self) -> str:
        return self.state.value

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'stats': dict(self.stats),
            }

    def reset(self):
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            self.success_count = 0
            logger.info(f"Circuit breaker {self.name} manually reset")


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception,
    name: Optional[str] = None,
):
    def decorator(func):
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=name or f"{func.__module__}.{func.__name__}",
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        wrapper.circuit_breaker = breaker  # type: ignore
        return wrapper

    return decorator


class CircuitBreakerRegistry:
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}

    def register(self, name: str, breaker: CircuitBreaker):
        self.breakers[name] = breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        return self.breakers.get(name)

    def get_all_stats(self) -> Dict[str, Dict]:
        return {n: b.get_stats() for n, b in self.breakers.items()}

    def reset_all(self):
        for b in self.breakers.values():
            b.reset()


circuit_breaker_registry = CircuitBreakerRegistry()

