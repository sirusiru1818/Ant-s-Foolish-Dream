"""Azure Blob Storage 연동 모듈"""
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import AzureError
import json
from typing import Optional, Dict, List
from src.config import settings
from src.encryption import EncryptionService


class BlobStorageService:
    """Azure Blob Storage 서비스"""
    
    def __init__(self):
        """Blob Storage 클라이언트 초기화"""
        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={settings.azure_storage_account_name};"
            f"AccountKey={settings.azure_storage_account_key};"
            f"EndpointSuffix=core.windows.net"
        )
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(
            settings.azure_storage_container_name
        )
        self.encryption_service = EncryptionService(settings.encryption_key)
        
        # 컨테이너가 없으면 생성
        try:
            self.container_client.create_container()
        except AzureError:
            pass  # 이미 존재하는 경우 무시
    
    def upload_text_file(self, blob_name: str, content: str, encrypt: bool = True) -> bool:
        """
        텍스트 파일 업로드
        
        Args:
            blob_name: Blob 이름 (경로 포함 가능)
            content: 업로드할 텍스트 내용
            encrypt: 암호화 여부
            
        Returns:
            성공 여부
        """
        try:
            if encrypt:
                content = self.encryption_service.encrypt(content)
            
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(content, overwrite=True)
            return True
        except AzureError as e:
            print(f"Blob 업로드 실패: {e}")
            return False
    
    def download_text_file(self, blob_name: str, decrypt: bool = True) -> Optional[str]:
        """
        텍스트 파일 다운로드
        
        Args:
            blob_name: Blob 이름
            decrypt: 복호화 여부
            
        Returns:
            파일 내용 또는 None
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            content = blob_client.download_blob().readall().decode('utf-8')
            
            if decrypt:
                content = self.encryption_service.decrypt(content)
            
            return content
        except AzureError as e:
            print(f"Blob 다운로드 실패: {e}")
            return None
    
    def upload_json(self, blob_name: str, data: Dict, encrypt: bool = True) -> bool:
        """JSON 데이터 업로드"""
        content = json.dumps(data, ensure_ascii=False, indent=2)
        return self.upload_text_file(blob_name, content, encrypt)
    
    def download_json(self, blob_name: str, decrypt: bool = True) -> Optional[Dict]:
        """JSON 데이터 다운로드"""
        content = self.download_text_file(blob_name, decrypt)
        if content:
            return json.loads(content)
        return None
    
    def list_blobs(self, prefix: str = "") -> List[str]:
        """Blob 목록 조회"""
        try:
            blobs = self.container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blobs]
        except AzureError as e:
            print(f"Blob 목록 조회 실패: {e}")
            return []
    
    def delete_blob(self, blob_name: str) -> bool:
        """Blob 삭제"""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            return True
        except AzureError as e:
            print(f"Blob 삭제 실패: {e}")
            return False
