# utils/config.py
import os
import toml
import streamlit as st
from pathlib import Path

# Streamlit 설정 파일 경로
STREAMLIT_CONFIG_DIR = ".streamlit"
STREAMLIT_CONFIG_FILE = "config.toml"
STREAMLIT_CONFIG_PATH = os.path.join(STREAMLIT_CONFIG_DIR, STREAMLIT_CONFIG_FILE)
CONFIG_PATH = Path(STREAMLIT_CONFIG_PATH)

# 기본 설정 값
DEFAULT_CONFIG = {
    "theme": {
        "primaryColor": "#ff4b4b",
        "backgroundColor": "#f0f2f6",
        "secondaryBackgroundColor": "#e0e0ef",
        "textColor": "#262730",
        "font": "sans serif",
    },
    "server": {
        "port": 8501,
        "enableCORS": False,
        "enableXsrfProtection": True,
        "maxUploadSize": 200,
    },
    "ocr_system": {
        "auto_start_detection": False,
    },
    "ui": {
        "auto_redirect_to_detection": False,
        "detection_page_path": "감지실행",
    },
    "roi": {
        "x": 100,
        "y": 100,
        "width": 200,
        "height": 100,
        "model_path": "yolov8n.pt",
    },
    "camera": {
        "width": 1024,
        "height": 768,
    },
}


def load_full_ui_config():
    """Streamlit config.toml 파일 전체를 로드합니다."""
    try:
        # 디렉토리가 없으면 생성
        os.makedirs(STREAMLIT_CONFIG_DIR, exist_ok=True)

        if os.path.exists(STREAMLIT_CONFIG_PATH):
            # 설정 파일 로드
            config = toml.load(STREAMLIT_CONFIG_PATH)
            # 기본값과 병합하여 누락된 키를 채움
            merged_config = {**DEFAULT_CONFIG, **config}
            for key in DEFAULT_CONFIG:
                if key not in merged_config:
                    merged_config[key] = DEFAULT_CONFIG[key]
            return merged_config
        else:
            # 설정 파일이 없으면 기본값 반환
            return DEFAULT_CONFIG
    except Exception as e:
        print(f"설정 파일 로드 중 오류 발생: {e}")
        return DEFAULT_CONFIG


def save_full_ui_config(config):
    """Streamlit config.toml 파일에 설정을 저장합니다."""
    try:
        # 디렉토리가 없으면 생성
        os.makedirs(STREAMLIT_CONFIG_DIR, exist_ok=True)

        # 설정 파일 저장
        with open(STREAMLIT_CONFIG_PATH, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        return True
    except Exception as e:
        print(f"설정 파일 저장 중 오류 발생: {e}")
        return False


def load_config(section):
    """
    설정 파일에서 특정 섹션을 로드합니다.
    Args:
        section (str): 로드할 섹션 이름
    Returns:
        dict: 섹션의 설정 값 (없으면 빈 딕셔너리 반환)
    """
    if CONFIG_PATH.exists():
        config = toml.load(CONFIG_PATH)
        return config.get(section, {})  # 섹션이 없으면 빈 딕셔너리 반환
    else:
        return {}  # 설정 파일이 없으면 빈 딕셔너리 반환


def save_config(section, cfg):
    """
    특정 섹션의 설정을 저장합니다.
    Args:
        section (str): 저장할 설정 섹션 이름 (예: "roi", "ocr_system").
        cfg (dict): 저장할 설정 값.
    Returns:
        bool: 저장 성공 여부.
    """
    try:
        # 현재 설정 로드
        config = load_full_ui_config()

        # 섹션 업데이트
        if section not in config:
            config[section] = {}

        config[section].update(cfg)

        # 설정 파일 저장
        return save_full_ui_config(config)
    except Exception as e:
        print(f"[CONFIG SAVE ERROR] {e}")
        return False
