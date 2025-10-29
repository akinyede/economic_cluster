"""Database initialization and utilities for KC Cluster Prediction Tool"""
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
import logging

from models import Base
from config import Config

logger = logging.getLogger(__name__)

class Database:
    """Database management class"""
    
    def __init__(self):
        self.config = Config()
        self.engine = None
        self.Session = None
        
    def initialize(self, use_sqlite=False):
        """Initialize database connection"""
        try:
            if use_sqlite or os.getenv('USE_SQLITE', 'false').lower() == 'true':
                # Use SQLite for testing/demo
                db_path = os.path.join(os.path.dirname(__file__), 'kc_clusters.db')
                self.engine = create_engine(f'sqlite:///{db_path}')
                logger.info(f"Using SQLite database at {db_path}")
            else:
                # Use PostgreSQL for production
                self.engine = create_engine(self.config.DATABASE_URL)
                logger.info("Connected to PostgreSQL database")
                
            # Create all tables
            Base.metadata.create_all(self.engine)

            # Lightweight migration: ensure new columns exist on SQLite
            try:
                if str(self.engine.url).startswith('sqlite'):  # simple runtime check
                    with self.engine.connect() as conn:
                        # Check if 'roi' column exists on 'clusters'
                        res = conn.execute(text("PRAGMA table_info('clusters')")).fetchall()
                        existing_cols = {row[1] for row in res} if res else set()
                        if 'roi' not in existing_cols:
                            conn.execute(text("ALTER TABLE clusters ADD COLUMN roi FLOAT"))
                            conn.commit()
                            logger.info("Applied lightweight migration: added 'roi' column to 'clusters'")
            except Exception as e:
                logger.warning(f"Schema migration check failed (non-fatal): {e}")
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            logger.info("Database initialized successfully")
            return True
            
        except OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            logger.info("Falling back to SQLite database")
            # Fallback to SQLite
            return self.initialize(use_sqlite=True)
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            return False
    
    def get_session(self):
        """Get a database session"""
        if not self.Session:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.Session()
    
    def reset_database(self):
        """Reset database (drop and recreate all tables)"""
        try:
            Base.metadata.drop_all(self.engine)
            Base.metadata.create_all(self.engine)
            logger.info("Database reset successfully")
            return True
        except Exception as e:
            logger.error(f"Database reset error: {e}")
            return False

# Global database instance
db = Database()
