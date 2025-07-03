from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class TiDBConfig(BaseModel):
    host: str = Field(description="TiDB host address")
    port: int = Field(4000, description="TiDB port number")
    user: str = Field(description="TiDB username")
    password: str = Field(description="TiDB password")
    database: str = Field(description="TiDB database name")
    collection_name: str = Field("mem0", description="Collection name (table name)")
    embedding_model_dims: Optional[int] = Field(1536, description="Dimensions of the embedding model")
    ssl_disabled: bool = Field(False, description="Whether to disable SSL connection")
    ssl_ca: Optional[str] = Field(None, description="Path to CA certificate file")
    ssl_cert: Optional[str] = Field(None, description="Path to client certificate file")
    ssl_key: Optional[str] = Field(None, description="Path to client private key file")
    ssl_verify_cert: bool = Field(True, description="Whether to verify server certificate")
    ssl_verify_identity: bool = Field(True, description="Whether to verify server identity")

    @model_validator(mode="before")
    def check_required_fields(cls, values):
        required_fields = ["host", "user", "password", "database"]
        missing_fields = [field for field in required_fields if not values.get(field)]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values