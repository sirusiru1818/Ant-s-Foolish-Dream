"""설정 관리 모듈"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # Azure OpenAI (선택사항 - 개발 환경에서 테스트 가능)
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_deployment_name: str = "gpt-4"
    
    # Azure Blob Storage (선택사항 - 개발 환경에서 테스트 가능)
    azure_storage_account_name: Optional[str] = None
    azure_storage_account_key: Optional[str] = None
    azure_storage_container_name: str = "stock-data"
    
    # Azure Machine Learning
    azure_ml_workspace_name: Optional[str] = None
    azure_ml_resource_group: Optional[str] = None
    azure_ml_subscription_id: Optional[str] = None
    
    # 암호화 (기본값 제공)
    encryption_key: str = "ZkxJX1lJcUlwZkVXQ21ELVdWX1FQY282Smh5VmdXRUYwR1M0X2RfWFJOUT0="
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
