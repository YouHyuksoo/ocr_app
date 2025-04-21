# app/services/camera.py 수정
import streamlit as st
import cv2
import time


def connect_camera(index, width, height, status_bar):
    """
    카메라를 연결하고 설정을 초기화합니다.
    Args:
        index (int): 웹캠 인덱스
        width (int): 카메라 해상도 너비
        height (int): 카메라 해상도 높이
        status_bar (StatusBar): 상태 표시 바 객체
    Returns:
        tuple: (cv2.VideoCapture 객체, 첫 번째 프레임)
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None, None

    # 카메라 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 첫 번째 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None, None

    status_bar.update("카메라 연결 성공")
    return cap, frame


def release_camera(status_bar):
    """
    카메라 리소스를 해제하고 세션 상태를 초기화합니다.
    """
    try:
        if "camera" in st.session_state and st.session_state.camera is not None:
            st.session_state.camera.release()  # 카메라 리소스 해제
            st.session_state.camera = None  # 세션 상태 초기화
            status_bar.update("카메라가 해제되었습니다.")
    except Exception as e:
        st.error(f"카메라 해제 중 오류 발생: {e}")
