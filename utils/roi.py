import cv2
import streamlit as st
import json
from streamlit_drawable_canvas import st_canvas


def draw_roi(frame, x, y, width, height):
    """녹색 사각형으로 ROI 시각화"""
    return cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)


def get_roi_from_canvas(canvas_result):
    """Streamlit canvas 결과에서 ROI 좌표 추출"""
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        obj = canvas_result.json_data["objects"][0]
        x = int(obj["left"])
        y = int(obj["top"])
        width = int(obj["width"])
        height = int(obj["height"])
        return x, y, width, height
    return None


def show_roi_setting_ui(cfg, key="canvas_roi"):
    """Sidebar ROI 설정 UI 표시 및 저장 처리"""
    st.sidebar.header("🟡 ROI 설정")
    st.sidebar.markdown("🖱️ 마우스로 ROI를 드래그하여 설정하세요.")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 255, 0.3)",
        stroke_width=2,
        stroke_color="#00FF00",
        background_color="#eee",
        update_streamlit=True,
        height=360,
        width=640,
        drawing_mode="rect",
        key=key,
    )

    roi_coords = get_roi_from_canvas(canvas_result)
    if roi_coords:
        x, y, w, h = roi_coords
        cfg.update({"x": x, "y": y, "width": w, "height": h})
        from utils.config import save_config

        save_config(cfg)
        st.sidebar.success("🟩 ROI 영역이 저장되었습니다.")
        return x, y, w, h

    return (
        cfg.get("x", 100),
        cfg.get("y", 100),
        cfg.get("width", 200),
        cfg.get("height", 100),
    )
