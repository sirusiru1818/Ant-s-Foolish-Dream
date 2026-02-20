"""애플리케이션 실행 스크립트"""
import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # 환경 변수로 reload 제어 (기본값: True)
    # 개발 중 venv 변경사항으로 인한 리로드를 피하려면:
    # RELOAD=false python run.py
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    if reload:
        # src와 scripts 디렉토리만 감시 (venv 제외)
        base_dir = Path(__file__).parent
        reload_dirs = [
            str(base_dir / "src"),
            str(base_dir / "scripts"),
        ]
    else:
        reload_dirs = None
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload,
        reload_dirs=reload_dirs
    )
