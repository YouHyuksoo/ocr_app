import streamlit as st
from utils.config import load_config, save_config
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide", page_title="ROI 설정", page_icon="📐")
st.title("📐 ROI 영역 설정")

# 설정 로드 및 기본값
roi_config = load_config("roi")
camera_config = load_config("camera")

# ROI 기본값
x = roi_config.get("x", 100)
y = roi_config.get("y", 100)
width = roi_config.get("width", 200)
height = roi_config.get("height", 100)

# 비디오 해상도 동기화 (카메라 설정에서 불러오기)
video_width = camera_config.get("width", 640)
video_height = camera_config.get("height", 480)

st.sidebar.header("🟡 ROI 설정")
st.sidebar.markdown("🖱️ 마우스로 ROI를 드래그하여 설정하세요.")

# 레이아웃: ROI 설정 화면만 배치
col_canvas = st

# "새로 지정" 버튼과 "ROI 설정 저장" 버튼을 한 줄에 배치
col1, col2 = st.columns(2)
with col1:
    if st.button("새로 지정"):
        # ROI 값을 초기화
        st.session_state.roi = {"x": 0, "y": 0, "width": 0, "height": 0}
        st.session_state.is_reset = True  # 새로 지정 상태로 변경
        st.session_state.save_message = ""  # 메시지 초기화

with col2:
    if st.button("ROI 설정 저장"):
        # ROI 설정 업데이트
        save_config("roi", st.session_state.roi)

        # 저장 메시지 표시
        st.session_state.save_message = "🟩 ROI 설정이 성공적으로 저장되었습니다."
        st.session_state.is_reset = False  # 저장 후 초기화 상태 해제

# 저장 메시지 표시
if "save_message" in st.session_state and st.session_state.save_message:
    if "성공적으로 저장되었습니다" in st.session_state.save_message:
        st.success(st.session_state.save_message)
    else:
        st.error(st.session_state.save_message)

# 기존 ROI 설정값을 캔버스에 반영
initial_drawing = {
    "version": "4.4.0",
    "objects": (
        [
            {
                "type": "rect",
                "left": x,
                "top": y,
                "width": width,
                "height": height,
                "fill": "rgba(0, 0, 255, 0.3)",
                "stroke": "#00FF00",
            }
        ]
        if not st.session_state.get("is_reset", False)
        else []
    ),  # 새로 지정 상태면 초기화
}

# ✅ 캔버스 생성
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 255, 0.3)",
    stroke_width=2,
    stroke_color="#00FF00",
    background_color="#eee",  # 기본 배경색 지정
    update_streamlit=True,
    height=video_height,  # 설정값에서 불러온 비디오 높이
    width=video_width,  # 설정값에서 불러온 비디오 너비
    drawing_mode="rect",
    initial_drawing=initial_drawing,  # 기존 ROI 설정값 반영
    key="canvas_roi",
)

# ROI 설정값 저장
if canvas_result.json_data and canvas_result.json_data["objects"]:
    obj = canvas_result.json_data["objects"][0]
    new_x = int(obj["left"])
    new_y = int(obj["top"])
    new_width = int(obj["width"])
    new_height = int(obj["height"])

    # ROI 값 업데이트
    st.session_state.roi = {
        "x": new_x,
        "y": new_y,
        "width": new_width,
        "height": new_height,
    }

# 초기 메시지 표시
if st.session_state.get("is_reset", False):
    st.info("🟡 ROI를 새로 지정한 후 저장 버튼을 클릭하세요.")
else:
    st.info("🟡 ROI를 설정한 후 저장 버튼을 클릭하세요.")

# 현재 설정값 표시
st.sidebar.markdown(
    f"""
    **현재 ROI 설정값:**
    - X: {st.session_state.roi['x']}
    - Y: {st.session_state.roi['y']}
    - 너비: {st.session_state.roi['width']}
    - 높이: {st.session_state.roi['height']}
    """
)
