"""데이터 암호화 모듈"""
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
from typing import Union


class EncryptionService:
    """데이터 암호화 서비스"""
    
    def __init__(self, key: str):
        """
        Args:
            key: 암호화 키 (32바이트 문자열 또는 base64 인코딩된 키)
        """
        # 키가 base64 인코딩되어 있으면 디코딩, 아니면 Fernet 키 생성
        try:
            # base64 디코딩 시도
            key_bytes = base64.urlsafe_b64decode(key)
            if len(key_bytes) != 32:
                raise ValueError("키 길이가 올바르지 않습니다")
            self.cipher = Fernet(base64.urlsafe_b64encode(key_bytes))
        except:
            # Fernet 키 생성
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'stock_encryption_salt',
                iterations=100000,
                backend=default_backend()
            )
            key_bytes = kdf.derive(key.encode())
            self.cipher = Fernet(base64.urlsafe_b64encode(key_bytes))
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """데이터 암호화"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        encrypted = self.cipher.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """데이터 복호화"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')
    
    @staticmethod
    def generate_key() -> str:
        """새로운 암호화 키 생성"""
        key = Fernet.generate_key()
        return base64.urlsafe_b64encode(key).decode('utf-8')
