# utils/config.py
import os
import toml
import streamlit as st
from pathlib import Path

# Streamlit 설정 파일 경로
STREAMLIT_CONFIG_DIR = ".streamlit"
STREAMLIT_CONFIG_FILE = "config.toml"
CUSTOM_CONFIG_FILE = "custom_config.toml"
STREAMLIT_CONFIG_PATH = os.path.join(STREAMLIT_CONFIG_DIR, STREAMLIT_CONFIG_FILE)
CUSTOM_CONFIG_PATH = os.path.join(STREAMLIT_CONFIG_DIR, CUSTOM_CONFIG_FILE)
CONFIG_PATH = Path(STREAMLIT_CONFIG_PATH)

# 기본 설정 값을 하나로 통합
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
    "paths": {
        "results_dir": "runs/detect",
        "dataset_dir": "dataset",
        "logs_dir": "logs",
    },
    "training": {
        "model_arch": "yolov8n.pt",
        "epochs": 100,
        "batch": 16,
        "imgsz": 640,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "project_name": "ocr_digit",
    },
}


def get_default_config(config_type):
    """지정된 타입의 기본 설정을 반환합니다."""
    return DEFAULT_CONFIG.get(config_type, {})


def get_streamlit_config_path():
    """Streamlit 설정 파일 경로를 반환합니다."""
    return os.path.join(Path.home(), ".streamlit", "config.toml")


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
    """커스텀 설정 파일에서 특정 섹션의 설정을 로드합니다."""
    try:
        if os.path.exists(CUSTOM_CONFIG_PATH):
            with open(CUSTOM_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = toml.load(f)
                return config.get(section, {})
    except Exception as e:
        print(f"설정 로드 오류: {e}")
        return {}


def save_config(section, settings):
    """커스텀 설정 파일에 특정 섹션의 설정을 저장합니다."""
    try:
        # 디렉토리가 없으면 생성
        os.makedirs(STREAMLIT_CONFIG_DIR, exist_ok=True)

        # 기존 설정 로드
        if os.path.exists(CUSTOM_CONFIG_PATH):
            with open(CUSTOM_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = toml.load(f)
        else:
            config = {}

        # 섹션이 없으면 생성
        if section not in config:
            config[section] = {}

        # 설정 업데이트
        config[section].update(settings)

        # 설정 저장
        with open(CUSTOM_CONFIG_PATH, "w", encoding="utf-8") as f:
            toml.dump(config, f)

        print(f"설정 저장 완료: {section}")
        return True
    except Exception as e:
        print(f"설정 저장 오류: {e}")
        return False
