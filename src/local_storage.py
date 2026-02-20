"""로컬 파일 스토리지 서비스"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, List
from src.config import settings
from src.encryption import EncryptionService


class LocalStorageService:
    """로컬 파일 스토리지 서비스"""
    
    def __init__(self, base_dir: str = "data"):
        """
        로컬 스토리지 초기화
        
        Args:
            base_dir: 데이터 저장 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.encryption_service = EncryptionService(settings.encryption_key)
    
    def _get_file_path(self, file_name: str) -> Path:
        """파일 경로 생성"""
        # 경로 구분자 처리
        file_name = file_name.replace("\\", "/")
        parts = file_name.split("/")
        
        # 디렉토리 생성
        file_path = self.base_dir
        for part in parts[:-1]:
            file_path = file_path / part
            file_path.mkdir(exist_ok=True)
        
        return file_path / parts[-1]
    
    def upload_text_file(self, file_name: str, content: str, encrypt: bool = True) -> bool:
        """
        텍스트 파일 저장
        
        Args:
            file_name: 파일 이름 (경로 포함 가능)
            content: 저장할 텍스트 내용
            encrypt: 암호화 여부
            
        Returns:
            성공 여부
        """
        try:
            if encrypt:
                content = self.encryption_service.encrypt(content)
            
            file_path = self._get_file_path(file_name)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        except Exception as e:
            print(f"파일 저장 실패: {e}")
            return False
    
    def download_text_file(self, file_name: str, decrypt: bool = True) -> Optional[str]:
        """
        텍스트 파일 읽기
        
        Args:
            file_name: 파일 이름
            decrypt: 복호화 여부
            
        Returns:
            파일 내용 또는 None
        """
        try:
            file_path = self._get_file_path(file_name)
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if decrypt:
                content = self.encryption_service.decrypt(content)
            
            return content
        except Exception as e:
            print(f"파일 읽기 실패: {e}")
            return None
    
    def upload_json(self, file_name: str, data: Dict, encrypt: bool = True) -> bool:
        """JSON 데이터 저장"""
        content = json.dumps(data, ensure_ascii=False, indent=2)
        return self.upload_text_file(file_name, content, encrypt)
    
    def download_json(self, file_name: str, decrypt: bool = True) -> Optional[Dict]:
        """JSON 데이터 읽기"""
        content = self.download_text_file(file_name, decrypt)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 실패: {e}")
                return None
        return None
    
    def list_files(self, prefix: str = "") -> List[str]:
        """
        파일 목록 조회
        
        Args:
            prefix: 파일 경로 접두사
            
        Returns:
            파일 목록
        """
        try:
            files = []
            prefix_path = self.base_dir / prefix if prefix else self.base_dir
            
            if not prefix_path.exists():
                return []
            
            for file_path in prefix_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.base_dir)
                    files.append(str(relative_path).replace("\\", "/"))
            
            return sorted(files)
        except Exception as e:
            print(f"파일 목록 조회 실패: {e}")
            return []
    
    def delete_file(self, file_name: str) -> bool:
        """파일 삭제"""
        try:
            file_path = self._get_file_path(file_name)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"파일 삭제 실패: {e}")
            return False
