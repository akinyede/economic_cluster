"""Memory optimization utilities for data processing"""
import gc
import os
import psutil
import logging
from typing import Iterator, Dict, List, Any, Optional
import pandas as pd
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        p = psutil.Process(os.getpid())
        mi = p.memory_info()
        return {
            'rss_mb': mi.rss / 1024 / 1024,
            'vms_mb': mi.vms / 1024 / 1024,
            'percent': p.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
        }

    @staticmethod
    def optimize_dataframe(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        start = df.memory_usage(deep=True).sum() / 1024 / 1024
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['object']).columns:
            try:
                nunique = df[col].nunique()
                if nunique / max(1, len(df[col])) < 0.5:
                    df[col] = df[col].astype('category')
            except Exception:
                pass
        end = df.memory_usage(deep=True).sum() / 1024 / 1024
        if verbose:
            logger.info(f"Mem reduced from {start:.1f}MB to {end:.1f}MB ({(1 - end/start) * 100:.1f}% reduction)")
        return df

    @staticmethod
    def process_large_file_in_chunks(filepath: str, processor_func: callable, chunk_size: int = 10000, **read_kwargs) -> List[Any]:
        results = []
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        logger.info(f"Processing file: {filepath} ({size_mb:.1f}MB)")
        idx = 0
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, **read_kwargs):
            chunk = MemoryOptimizer.optimize_dataframe(chunk)
            results.append(processor_func(chunk))
            del chunk
            idx += 1
            if idx % 10 == 0:
                gc.collect()
                mu = MemoryOptimizer.get_memory_usage()
                logger.debug(f"Processed {idx} chunks; RSS={mu['rss_mb']:.1f}MB")
        return results

    @staticmethod
    @contextmanager
    def memory_limit(max_memory_mb: int):
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_mb * 1024 * 1024, hard))
        try:
            yield
        finally:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

    @staticmethod
    def batch_generator(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    @staticmethod
    def estimate_dataframe_memory(rows: int, columns: Dict[str, str]) -> float:
        dtype_sizes = {
            'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8,
            'uint8': 1, 'uint16': 2, 'uint32': 4, 'uint64': 8,
            'float16': 2, 'float32': 4, 'float64': 8,
            'bool': 1,
            'datetime64': 8,
            'timedelta64': 8,
            'category': 2,
            'object': 50,
        }
        total = 0
        for _, dtype in columns.items():
            ds = next((size for key, size in dtype_sizes.items() if key in str(dtype)), dtype_sizes['object'])
            total += rows * ds
        return total / 1024 / 1024


memory_optimizer = MemoryOptimizer()

