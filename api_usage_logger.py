"""
MongoDB API usage logger สำหรับ ai-predict-occupation-main-sub

แยก collection ตาม success/error:
- ai_predict_occupation_main_sub_log         — success
- ai_predict_occupation_main_sub_log_error   — error
"""

from datetime import datetime
from typing import Optional, Dict
from enum import Enum
import logging
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MONGODB_CONFIG = {
    'host': os.getenv("MONGO_HOST", "10.100.100.215"),
    'port': int(os.getenv("MONGO_PORT", "27017")),
    'username': os.getenv("MONGO_USER"),
    'password': os.getenv("MONGO_PASS"),
    'database': os.getenv("MONGO_DB", "mydb"),
    'auth_source': os.getenv("MONGO_AUTH_SOURCE", "mydb"),
}

COLLECTION_SUCCESS = 'ai_predict_occupation_main_sub_log'
COLLECTION_ERROR = 'ai_predict_occupation_main_sub_log_error'


class EndpointType(str, Enum):
    PREDICT_MEMBER = "predict-member"
    PREDICT_EMPLOYER = "predict-employer"


class MongoDBClient:
    _instance = None
    _client = None
    _db = None
    _executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="mongo_log_")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connect()
        return cls._instance

    def _connect(self):
        if not MONGODB_CONFIG.get('username') or not MONGODB_CONFIG.get('password'):
            logger.warning(
                "MongoDB credentials not set (MONGO_USER/MONGO_PASS env vars) — "
                "API logging disabled, service จะรันต่อปกติ"
            )
            self._client = None
            self._db = None
            return
        try:
            uri = (
                f"mongodb://{MONGODB_CONFIG['username']}:{MONGODB_CONFIG['password']}"
                f"@{MONGODB_CONFIG['host']}:{MONGODB_CONFIG['port']}"
                f"/{MONGODB_CONFIG['database']}"
                f"?authSource={MONGODB_CONFIG['auth_source']}&directConnection=true"
            )
            self._client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
            self._client.admin.command('ping')
            self._db = self._client[MONGODB_CONFIG['database']]
            self._create_indexes()
            logger.info("✓ MongoDB connected for API logging")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB connection failed: {e}")
            self._client = None
            self._db = None
        except Exception as e:
            logger.error(f"MongoDB error: {e}")
            self._client = None
            self._db = None

    def _create_indexes(self):
        if self._db is None:
            return
        try:
            for name in (COLLECTION_SUCCESS, COLLECTION_ERROR):
                coll = self._db[name]
                coll.create_index([("endpoint_type", ASCENDING)])
                coll.create_index([("timestamp", DESCENDING)])
                coll.create_index([("ip_address", ASCENDING)])
                coll.create_index([("date", ASCENDING)])
            logger.info("✓ MongoDB indexes created")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")

    def get_collection(self, success: bool = True):
        if self._db is not None:
            return self._db[COLLECTION_SUCCESS if success else COLLECTION_ERROR]
        return None

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._db is not None


class APIUsageLogger:
    def __init__(self):
        self.mongo = MongoDBClient()

    def log_sync(
        self,
        endpoint_type: EndpointType,
        ip_address: str = "unknown",
        processing_time_ms: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        extra_data: Optional[Dict] = None,
    ) -> bool:
        if not self.mongo.is_connected:
            return False
        coll = self.mongo.get_collection(success)
        if coll is None:
            return False
        try:
            now = datetime.utcnow()
            doc = {
                "endpoint_type": endpoint_type.value,
                "ip_address": ip_address,
                "timestamp": now,
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "hour": now.hour,
                "day_of_week": now.weekday(),
                "processing_time_ms": processing_time_ms,
                "success": success,
                "error_message": error_message,
            }
            if extra_data:
                doc["extra"] = extra_data
            coll.insert_one(doc)
            return True
        except Exception as e:
            logger.error(f"Error saving API log: {e}")
            return False

    def log_fire_and_forget(self, endpoint_type: EndpointType, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.run_in_executor(self.mongo._executor, lambda: self.log_sync(endpoint_type, **kwargs))
            else:
                self.log_sync(endpoint_type, **kwargs)
        except Exception as e:
            logger.error(f"Error in fire-and-forget log: {e}")


api_logger = APIUsageLogger()
