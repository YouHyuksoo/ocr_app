# components/sidebar.py
import streamlit as st
import torch
from utils.config import load_config, save_config


def setup_sidebar(status_bar):
    """
    Streamlit 사이드바 UI 설정

    Returns:
        tuple: (비디오 소스, 감지 신뢰도, 숫자 감지 모드, PLC 설정)
    """
    st.sidebar.header("🎥 비디오 소스")

    # GPU 사용 여부 표시
    if torch.cuda.is_available():
        st.sidebar.markdown(f"**🟢 CUDA 사용 중:** {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.markdown("**🟡 CPU 사용 중 (CUDA 미사용)**")

    # 설정 파일 로드
    camera_config = load_config("camera")  # 카메라 설정 로드
    training_config = load_config("training")  # 학습 설정 로드

    # 비디오 해상도 동기화
    video_width = camera_config.get("width", 640)
    video_height = camera_config.get("height", 480)

    # 모델 경로 로드
    model_path = training_config.get("model_path", "yolov8n.pt")

    # 비디오 소스 선택
    video_source = st.sidebar.selectbox(
        "입력 소스 선택",
        ["웹캠", "비디오 파일", "이미지"],
        index=0,
    )

    # 감지 설정 UI 생성
    st.sidebar.header("🔍 감지 설정")

    # 설정 파일에서 감지 설정 로드
    detection_config = load_config("detection") or {}

    # 기본값 설정
    default_confidence = detection_config.get("confidence_threshold", 0.5)
    default_mode = detection_config.get("mode_option", "숫자 감지")

    # 감지 설정 UI
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.1, 1.0, default_confidence, 0.05
    )
    mode_option = st.sidebar.radio(
        "감지 모드 선택",
        ["숫자 감지", "전체 객체 감지"],
        index=0 if default_mode == "숫자 감지" else 1,
    )

    # 설정 변경 시 저장
    if st.sidebar.button("설정 저장"):
        save_config(
            "detection",
            {
                "confidence_threshold": confidence_threshold,
                "mode_option": mode_option,
            },
        )
        st.sidebar.success("설정이 저장되었습니다.")

    is_digit_mode = mode_option == "숫자 감지"

    # PLC 설정 UI 생성
    st.sidebar.header("🔄 PLC 설정")
    plc_settings = {
        "enabled": st.sidebar.checkbox("PLC 전송", value=True),
        "ip": st.sidebar.text_input("PLC IP", "192.168.0.10"),
        "port": st.sidebar.number_input("포트", value=502),
        "register": st.sidebar.number_input("레지스터", value=100),
        "retry": st.sidebar.number_input("재시도 횟수", 0, 5, 2),
    }

    return video_source, confidence_threshold, is_digit_mode, plc_settings
