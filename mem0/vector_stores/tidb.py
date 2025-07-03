import json
import logging
from typing import List, Optional, Dict, Any
import uuid

from pydantic import BaseModel

try:
    import pymysql
except ImportError:
    raise ImportError("The 'pymysql' library is required. Please install it using 'pip install pymysql'.")

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[dict]


class TiDB(VectorStoreBase):
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        collection_name: str,
        embedding_model_dims: int,
        ssl_disabled: bool = False,
        ssl_ca: str = None,
        ssl_cert: str = None,
        ssl_key: str = None,
        ssl_verify_cert: bool = True,
        ssl_verify_identity: bool = True,
    ):
        """
        Initialize the TiDB vector store.

        Args:
            host (str): TiDB host address
            port (int): TiDB port
            user (str): TiDB username
            password (str): TiDB password
            database (str): TiDB database name
            collection_name (str): Collection name (table name)
            embedding_model_dims (int): Dimension of the embedding vector
            ssl_disabled (bool): Whether to disable SSL
            ssl_ca (str): Path to CA certificate file
            ssl_cert (str): Path to client certificate file
            ssl_key (str): Path to client key file
            ssl_verify_cert (bool): Whether to verify server certificate
            ssl_verify_identity (bool): Whether to verify server identity
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        
        # SSL configuration
        self.ssl_config = {}
        if not ssl_disabled:
            if ssl_ca:
                self.ssl_config['ca'] = ssl_ca
            if ssl_cert:
                self.ssl_config['cert'] = ssl_cert
            if ssl_key:
                self.ssl_config['key'] = ssl_key
            self.ssl_config['check_hostname'] = ssl_verify_identity
            self.ssl_config['verify_mode'] = ssl_verify_cert

        self.conn = self._create_connection()
        self.cur = self.conn.cursor()

        # Create collection if it doesn't exist
        collections = self.list_cols()
        if collection_name not in collections:
            self.create_col(embedding_model_dims)

    def _create_connection(self):
        """Create a connection to TiDB."""
        config = {
            'host': self.host,
            'port': self.port,
            'user': self.user,
            'password': self.password,
            'database': self.database,
            'charset': 'utf8mb4',
            'autocommit': True,
        }
        
        if self.ssl_config:
            config['ssl'] = self.ssl_config
            
        return pymysql.connect(**config)

    def create_col(self, vector_size: int, distance: str = 'cosine'):
        """
        Create a new collection (table in TiDB).

        Args:
            vector_size (int): Dimension of the embedding vector
            distance (str): Distance metric for vector similarity ('cosine', 'l2', 'inner_product')
        """
        logger.info(f"Creating collection {self.collection_name} with vector size {vector_size}")
        
        # Create table with vector column
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{self.collection_name}` (
            id VARCHAR(36) PRIMARY KEY,
            vector JSON NOT NULL,
            payload JSON,
            INDEX idx_vector ((CAST(vector AS CHAR(65535) ARRAY)))
        )
        """
        
        self.cur.execute(create_table_sql)
        logger.info(f"Collection {self.collection_name} created successfully")

    def insert(self, vectors: List[List[float]], payloads: List[Dict] = None, ids: List[str] = None):
        """
        Insert vectors into a collection.

        Args:
            vectors (List[List[float]]): List of vectors to insert
            payloads (List[Dict], optional): List of payloads corresponding to vectors
            ids (List[str], optional): List of IDs corresponding to vectors
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        
        if not vectors:
            return
            
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        if payloads is None:
            payloads = [{}] * len(vectors)

        # Prepare data for insertion
        data = []
        for i, vector in enumerate(vectors):
            data.append((
                ids[i],
                json.dumps(vector),
                json.dumps(payloads[i])
            ))

        # Insert data
        insert_sql = f"""
        INSERT INTO `{self.collection_name}` (id, vector, payload) 
        VALUES (%s, %s, %s)
        """
        
        self.cur.executemany(insert_sql, data)
        logger.info(f"Successfully inserted {len(vectors)} vectors")

    def search(self, query: str, vectors: List[float], limit: int = 5, filters: Dict = None) -> List[OutputData]:
        """
        Search for similar vectors using TiDB's vector search capabilities.

        Args:
            query (str): Query string (not used in vector search)
            vectors (List[float]): Query vector
            limit (int): Number of results to return
            filters (Dict): Filters to apply to the search

        Returns:
            List[OutputData]: Search results
        """
        logger.info(f"Searching for similar vectors in collection {self.collection_name}")
        
        # Build the search query using TiDB's vector search function
        vector_json = json.dumps(vectors)
        
        # Base query with vector similarity search
        base_query = f"""
        SELECT id, 
               VEC_COSINE_DISTANCE(JSON_EXTRACT(vector, '$'), JSON_EXTRACT(%s, '$')) AS distance,
               payload
        FROM `{self.collection_name}`
        """
        
        query_params = [vector_json]
        
        # Add filters if provided
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                filter_conditions.append(f"JSON_EXTRACT(payload, '$.{key}') = %s")
                query_params.append(str(value))
            
            if filter_conditions:
                base_query += " WHERE " + " AND ".join(filter_conditions)
        
        # Add ordering and limit
        base_query += f" ORDER BY distance LIMIT %s"
        query_params.append(limit)
        
        self.cur.execute(base_query, query_params)
        results = self.cur.fetchall()
        
        return [
            OutputData(
                id=str(row[0]),
                score=float(row[1]),
                payload=json.loads(row[2]) if row[2] else {}
            )
            for row in results
        ]

    def delete(self, vector_id: str):
        """
        Delete a vector by ID.

        Args:
            vector_id (str): ID of the vector to delete
        """
        logger.info(f"Deleting vector with ID {vector_id}")
        
        delete_sql = f"DELETE FROM `{self.collection_name}` WHERE id = %s"
        self.cur.execute(delete_sql, (vector_id,))
        
        logger.info(f"Successfully deleted vector with ID {vector_id}")

    def update(self, vector_id: str, vector: List[float] = None, payload: Dict = None):
        """
        Update a vector and its payload.

        Args:
            vector_id (str): ID of the vector to update
            vector (List[float], optional): Updated vector
            payload (Dict, optional): Updated payload
        """
        logger.info(f"Updating vector with ID {vector_id}")
        
        updates = []
        params = []
        
        if vector is not None:
            updates.append("vector = %s")
            params.append(json.dumps(vector))
        
        if payload is not None:
            updates.append("payload = %s")
            params.append(json.dumps(payload))
        
        if not updates:
            return
        
        params.append(vector_id)
        update_sql = f"UPDATE `{self.collection_name}` SET {', '.join(updates)} WHERE id = %s"
        
        self.cur.execute(update_sql, params)
        logger.info(f"Successfully updated vector with ID {vector_id}")

    def get(self, vector_id: str) -> Optional[OutputData]:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (str): ID of the vector to retrieve

        Returns:
            Optional[OutputData]: Retrieved vector or None if not found
        """
        logger.info(f"Retrieving vector with ID {vector_id}")
        
        select_sql = f"SELECT id, vector, payload FROM `{self.collection_name}` WHERE id = %s"
        self.cur.execute(select_sql, (vector_id,))
        
        result = self.cur.fetchone()
        if not result:
            return None
        
        return OutputData(
            id=str(result[0]),
            score=None,
            payload=json.loads(result[2]) if result[2] else {}
        )

    def list_cols(self) -> List[str]:
        """
        List all collections (tables).

        Returns:
            List[str]: List of collection names
        """
        self.cur.execute("SHOW TABLES")
        return [row[0] for row in self.cur.fetchall()]

    def delete_col(self):
        """Delete a collection (table)."""
        logger.warning(f"Deleting collection {self.collection_name}")
        
        drop_sql = f"DROP TABLE IF EXISTS `{self.collection_name}`"
        self.cur.execute(drop_sql)
        
        logger.info(f"Successfully deleted collection {self.collection_name}")

    def col_info(self) -> Dict[str, Any]:
        """
        Get information about a collection.

        Returns:
            Dict[str, Any]: Collection information
        """
        # Get table information
        info_sql = f"""
        SELECT 
            TABLE_NAME,
            TABLE_ROWS,
            DATA_LENGTH,
            INDEX_LENGTH
        FROM information_schema.TABLES 
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        """
        
        self.cur.execute(info_sql, (self.database, self.collection_name))
        result = self.cur.fetchone()
        
        if not result:
            return {}
        
        return {
            "name": result[0],
            "count": result[1] or 0,
            "data_size": result[2] or 0,
            "index_size": result[3] or 0,
            "total_size": (result[2] or 0) + (result[3] or 0)
        }

    def list(self, filters: Dict = None, limit: int = 100) -> List[List[OutputData]]:
        """
        List all vectors in a collection.

        Args:
            filters (Dict, optional): Filters to apply to the list
            limit (int): Number of vectors to return

        Returns:
            List[List[OutputData]]: List of vectors
        """
        logger.info(f"Listing vectors from collection {self.collection_name}")
        
        base_query = f"SELECT id, vector, payload FROM `{self.collection_name}`"
        query_params = []
        
        # Add filters if provided
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                filter_conditions.append(f"JSON_EXTRACT(payload, '$.{key}') = %s")
                query_params.append(str(value))
            
            if filter_conditions:
                base_query += " WHERE " + " AND ".join(filter_conditions)
        
        base_query += f" LIMIT %s"
        query_params.append(limit)
        
        self.cur.execute(base_query, query_params)
        results = self.cur.fetchall()
        
        return [[
            OutputData(
                id=str(row[0]),
                score=None,
                payload=json.loads(row[2]) if row[2] else {}
            )
            for row in results
        ]]

    def reset(self):
        """Reset the collection by deleting and recreating it."""
        logger.warning(f"Resetting collection {self.collection_name}")
        self.delete_col()
        self.create_col(self.embedding_model_dims)
        logger.info(f"Successfully reset collection {self.collection_name}")

    def __del__(self):
        """Close the database connection when the object is deleted."""
        if hasattr(self, 'cur') and self.cur:
            self.cur.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()