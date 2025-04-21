# 리팩토링된 Streamlit OCR 실시간 대시보드 (감지 이미지에 바운딩 박스 포함 처리 개선)
import streamlit as st
import atexit
import torch
from datetime import datetime
from PIL import Image
import cv2
from collections import deque
import numpy as np
import base64
from io import BytesIO

# 모듈 가져오기
from utils.config import load_config
from utils.logger import log_detection, log_plc_transmission
from components.status import StatusBar
from components.sidebar import setup_sidebar
from services.detector import setup_detector, process_detections
from services.plc import send_to_plc
from services.snapshot import save_snapshot
from utils.roi import draw_roi
from services.camera import connect_camera, release_camera

import debugpy

st.set_page_config(layout="wide", page_title="OCR 실시간 대시보드", page_icon="📸")
st.title("📸 차량번호 검출시스템")
status_bar = StatusBar()

# if "debug_listening" not in st.session_state:
#     try:
#         debugpy.listen(5678)
#         st.session_state.debug_listening = True
#         st.write("🧩 디버거 연결 대기 중...")
#         debugpy.wait_for_client()
#         st.success("🐞 디버거가 연결되었습니다!")
#     except RuntimeError:
#         st.warning("디버거가 이미 연결되어 있거나 포트가 사용 중입니다.")


# 앱 종료 시 리소스 정리 함수
def cleanup():
    if "camera" in st.session_state and st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
        print("카메라 리소스가 해제되었습니다.")


atexit.register(cleanup)

ocr_config = load_config("ocr_system") or {}
roi_config = load_config("roi") or {}
camera_config = load_config("camera") or {}
detection_config = load_config("detection") or {}

# 설정값
auto_start = ocr_config.get("auto_start_detection", False)
model_path = detection_config.get("model_path", "yolov8n.pt")
camera_width = camera_config.get("width", 320)
camera_height = camera_config.get("height", 240)

# 세션 상태 초기화
st.session_state.setdefault("detecting", auto_start)
st.session_state.setdefault("need_camera_connection", auto_start)
st.session_state.setdefault("auto_start_done", False)
st.session_state.setdefault("camera", None)
st.session_state.setdefault("start_button_clicked", False)

# setup_sidebar 함수 호출
video_source, confidence_threshold, is_digit_mode, plc_settings = setup_sidebar(
    status_bar
)

# 이미지 업로드 UI
uploaded_image = None
if video_source == "이미지":
    uploaded_image = st.sidebar.file_uploader(
        "🖼️ 이미지 업로드", type=["jpg", "png"], help="이미지를 업로드하세요."
    )

# 모델 초기화
model = setup_detector(model_path)
recent_logs = deque(maxlen=100)
last_sent_digit = None
sent_history = deque(maxlen=10)

# 버튼 및 이미지 확대/축소 컨테이너 (항상 상단에 위치하도록 고정)
with st.container():
    st.markdown("### 제어 버튼 및 이미지 크기 제어")
    start_col, stop_col, release_col, zoom_out_col, zoom_in_col = st.columns(5)
    start_button = start_col.button("▶️ 감지 시작")
    stop_button = stop_col.button("⏹️ 감지 중지")
    release_button = release_col.button("🔌 카메라 끄기")
    zoom_out = zoom_out_col.button("➖ 작게")
    zoom_in = zoom_in_col.button("➕ 크게")

    if "image_width_percent" not in st.session_state:
        st.session_state.image_width_percent = 40

    if zoom_out and st.session_state.image_width_percent > 10:
        st.session_state.image_width_percent -= 10
    if zoom_in and st.session_state.image_width_percent < 100:
        st.session_state.image_width_percent += 10

if start_button:
    st.session_state.detecting = True
    st.session_state.start_button_clicked = True
    st.session_state.need_camera_connection = video_source == "웹캠"

if stop_button:
    st.session_state.detecting = False
    status_bar.update("감지 중지됨")
    cv2.destroyAllWindows()

if release_button:
    if st.session_state.camera:
        release_camera(status_bar)
        st.session_state.camera = None
        st.success("카메라 연결 해제 완료")
    else:
        st.warning("연결된 카메라가 없습니다.")

# 감지 처리 로직
with st.container():
    FRAME_WINDOW = st.empty()
    LOG_WINDOW = st.empty()

    if st.session_state.detecting:
        if video_source == "이미지":
            if uploaded_image:
                file_bytes = np.asarray(
                    bytearray(uploaded_image.read()), dtype=np.uint8
                )
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                try:
                    status_bar.update("이미지에서 객체 감지 중...")
                    detections, detected_digits, combined, annotated_display = (
                        process_detections(
                            model,
                            frame,
                            confidence_threshold,
                            is_digit_mode,
                            {},
                            status_bar,
                        )
                    )
                    if is_digit_mode and len(detected_digits) == 3:
                        combined = "".join(detected_digits)
                        st.sidebar.markdown(f"**감지된 숫자:** `{combined}`")

                    buffered = BytesIO()
                    Image.fromarray(
                        cv2.cvtColor(annotated_display, cv2.COLOR_BGR2RGB)
                    ).save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()

                    FRAME_WINDOW.markdown(
                        f"<img src='data:image/png;base64,{img_base64}' style='width:{st.session_state.image_width_percent}%;'/>",
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"이미지 감지 중 오류 발생: {e}")
                    status_bar.update("오류 발생. 감지 중단.")

        elif video_source == "웹캠":
            if st.session_state.need_camera_connection:
                if not isinstance(st.session_state.camera, cv2.VideoCapture):
                    cap, frame = connect_camera(
                        0, camera_width, camera_height, status_bar
                    )
                    if isinstance(cap, cv2.VideoCapture):
                        st.session_state.camera = cap
                        st.success("카메라 연결 성공")
                    else:
                        st.error("카메라 연결 실패")
                        st.session_state.detecting = False
                st.session_state.need_camera_connection = False

            if st.session_state.camera:
                while st.session_state.detecting:
                    ret, frame = st.session_state.camera.read()
                    if not ret:
                        st.warning("프레임 읽기 실패")
                        break

                    detections, detected_digits, combined, annotated_display = (
                        process_detections(
                            model,
                            frame,
                            confidence_threshold,
                            is_digit_mode,
                            {},
                            status_bar,
                        )
                    )

                    buffered = BytesIO()
                    Image.fromarray(
                        cv2.cvtColor(annotated_display, cv2.COLOR_BGR2RGB)
                    ).save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()

                    FRAME_WINDOW.markdown(
                        f"<img src='data:image/png;base64,{img_base64}' style='width:{st.session_state.image_width_percent}%;'/>",
                        unsafe_allow_html=True,
                    )

st.markdown("---")
st.markdown("### 카메라 설정값")
st.write(f"- **카메라 해상도**: {camera_width} x {camera_height}")
st.write(f"- **모델 경로**: {model_path}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        torch.cuda.set_per_process_memory_fraction(0.7, device)

if not st.session_state.detecting and st.session_state.camera:
    release_camera(status_bar)
    st.session_state.camera = None
