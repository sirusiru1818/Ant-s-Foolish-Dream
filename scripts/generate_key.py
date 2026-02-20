"""암호화 키 생성 스크립트"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cryptography.fernet import Fernet
import base64

if __name__ == "__main__":
    # Fernet 키 생성
    key = Fernet.generate_key()
    key_str = base64.urlsafe_b64encode(key).decode('utf-8')
    
    print("\n" + "="*50)
    print("생성된 암호화 키:")
    print("="*50)
    print(key_str)
    print("="*50)
    print("\n이 키를 .env 파일의 ENCRYPTION_KEY에 설정하세요.")
    print("주의: 이 키를 안전하게 보관하세요. 키를 잃어버리면 암호화된 데이터를 복호화할 수 없습니다.\n")
