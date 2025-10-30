"""Database initialization and utilities for KC Cluster Prediction Tool (with pooling)"""
import os
import logging
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool, QueuePool

from models import Base
from config import Config

logger = logging.getLogger(__name__)


class Database:
    """Database management class with connection pooling and scoped sessions"""

    def __init__(self):
        self.config = Config()
        self.engine = None
        self.SessionFactory = None
        self.scoped = None
        self._is_sqlite = False

    def initialize(self, use_sqlite: bool = False) -> bool:
        """Initialize database connection and session registry"""
        try:
            # Decide backend
            self._is_sqlite = bool(use_sqlite or os.getenv('USE_SQLITE', 'false').lower() == 'true')
            if self._is_sqlite:
                db_path = os.path.join(os.path.dirname(__file__), 'kc_clusters.db')
                self.engine = create_engine(
                    f'sqlite:///{db_path}',
                    connect_args={'check_same_thread': False, 'timeout': 30},
                    poolclass=NullPool,
                    echo=False,
                )
                logger.info(f"Using SQLite database at {db_path}")

                # Optimize SQLite pragmas
                @event.listens_for(self.engine, "connect")
                def _set_sqlite_pragmas(dbapi_conn, connection_record):
                    try:
                        cur = dbapi_conn.cursor()
                        cur.execute("PRAGMA cache_size = -64000")
                        cur.execute("PRAGMA temp_store = MEMORY")
                        cur.execute("PRAGMA mmap_size = 268435456")
                        cur.execute("PRAGMA synchronous = NORMAL")
                        cur.execute("PRAGMA journal_mode = WAL")
                        cur.execute("PRAGMA busy_timeout = 30000")
                        cur.execute("PRAGMA foreign_keys = ON")
                        cur.close()
                    except Exception:
                        pass
            else:
                # PostgreSQL with QueuePool
                db_url = os.getenv('DATABASE_URL', self.config.DATABASE_URL)
                self.engine = create_engine(
                    db_url,
                    poolclass=QueuePool,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False,
                    connect_args={'connect_timeout': 10},
                )
                logger.info("Connected to PostgreSQL database")

            # Create tables if needed
            Base.metadata.create_all(self.engine)

            # Lightweight migration for ROI column on SQLite
            try:
                if str(self.engine.url).startswith('sqlite'):
                    with self.engine.connect() as conn:
                        res = conn.execute(text("PRAGMA table_info('clusters')")).fetchall()
                        cols = {row[1] for row in res} if res else set()
                        if 'roi' not in cols:
                            conn.execute(text("ALTER TABLE clusters ADD COLUMN roi FLOAT"))
                            conn.commit()
                            logger.info("Applied migration: added 'roi' column to 'clusters'")
            except Exception as e:
                logger.warning(f"Schema migration check failed (non-fatal): {e}")

            # Session factory + scoped registry (thread-safe)
            self.SessionFactory = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)
            self.scoped = scoped_session(self.SessionFactory)

            logger.info("Database initialized successfully")
            return True

        except OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            logger.info("Falling back to SQLite database")
            return self.initialize(use_sqlite=True)
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            return False

    def get_session(self):
        """Get a thread-local scoped session"""
        if not self.scoped:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.scoped()

    def reset_database(self) -> bool:
        """Reset database (drop and recreate all tables)"""
        try:
            Base.metadata.drop_all(self.engine)
            Base.metadata.create_all(self.engine)
            logger.info("Database reset successfully")
            return True
        except Exception as e:
            logger.error(f"Database reset error: {e}")
            return False

    def close_all_sessions(self):
        try:
            if self.scoped:
                self.scoped.remove()
        except Exception:
            pass


# Global database instance
db = Database()
