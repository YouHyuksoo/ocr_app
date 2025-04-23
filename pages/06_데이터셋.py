from utils.datasetgen import generate_dataset, zip_dataset_folder, make_image

LAST_ZIP_BYTES = None
LAST_ZIP_META = None
import streamlit as st
import os
import tempfile
import shutil
from utils.config import load_config, save_config
from PIL import Image
import glob
import time

st.set_page_config(page_title="📦 데이터셋 생성기", layout="wide")
st.title("🧪 YOLO 학습용 숫자 데이터셋 생성기")


# RGB 리스트나 hex 문자열을 RGB 튜플로 변환하는 함수들
def rgb_to_hex(rgb_value):
    """RGB 값을 hex 문자열로 변환합니다."""
    if isinstance(rgb_value, (list, tuple)) and len(rgb_value) == 3:
        return "#{:02x}{:02x}{:02x}".format(*rgb_value)
    return "#FFFFFF"  # 기본값


def hex_to_rgb(hex_str):
    """hex 문자열을 RGB 튜플로 변환합니다."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))


def ensure_rgb_tuple(color):
    """색상값이 RGB 튜플인지 확인하고 변환합니다."""
    if isinstance(color, str):
        return hex_to_rgb(color)
    elif isinstance(color, (list, tuple)) and len(color) == 3:
        return tuple(color)
    return (255, 255, 255)  # 기본값


# Load config from YAML
cfg = load_config("dataset")

# --- Sidebar 입력 폼 ---
st.sidebar.header("📋 데이터셋 조건 설정")
with st.sidebar.form("dataset_form"):
    digits = st.text_input(
        "사용할 숫자 목록 (예: 0,1,2,3,4)",
        value=",".join(cfg.get("digits", [str(i) for i in range(10)])),
    )
    length = st.slider("자릿수 (예: 3자리)", 1, 10, value=cfg.get("length", 3))
    image_size = st.text_input(
        "이미지 크기 (width,height)",
        value=",".join(map(str, cfg.get("image_size", [640, 320]))),
    )
    digit_box_size = st.text_input(
        "숫자 영역 크기 (width,height)",
        value=",".join(map(str, cfg.get("digit_box_size", [180, 120]))),
    )
    digit_spacing = st.number_input(
        "숫자 간격 (px)", min_value=0, value=cfg.get("digit_spacing", 20)
    )

    # 컬러 피커 설정
    background_hex = rgb_to_hex(cfg.get("background", [255, 255, 255]))
    text_color_hex = rgb_to_hex(cfg.get("text_color", [0, 0, 0]))

    background = st.color_picker(
        "배경색", value=background_hex, help="이미지의 배경색을 선택합니다."
    )

    text_color = st.color_picker(
        "텍스트 색상", value=text_color_hex, help="숫자의 색상을 선택합니다."
    )

    # 폰트 선택
    available_fonts = sorted(
        glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    ) + sorted(glob.glob("C:/Windows/Fonts/*.ttf"))
    font_path = st.selectbox(
        "폰트 선택",
        available_fonts,
        index=(
            available_fonts.index(cfg.get("font_path", "arial.ttf"))
            if cfg.get("font_path", "arial.ttf") in available_fonts
            else 0
        ),
    )

    apply_distortion = st.checkbox(
        "랜덤 왜곡 (회전, 기울임 등)", value=cfg.get("apply_distortion", False)
    )
    random_position = st.checkbox(
        "숫자 위치 무작위화", value=cfg.get("random_position", False)
    )
    use_random_bg = st.checkbox(
        "랜덤 배경 이미지 사용", value=cfg.get("use_random_bg", False)
    )
    apply_blur = st.checkbox("글자 흐림 효과 적용", value=cfg.get("apply_blur", False))
    apply_noise = st.checkbox("노이즈 효과 추가", value=cfg.get("apply_noise", False))

    submitted = st.form_submit_button("💾 조건 저장")
    if submitted:
        try:
            config_to_save = {
                "digits": [d.strip() for d in digits.split(",")],
                "length": length,
                "image_size": list(map(int, image_size.split(","))),
                "digit_box_size": list(map(int, digit_box_size.split(","))),
                "digit_spacing": digit_spacing,
                "background": ensure_rgb_tuple(background),
                "text_color": ensure_rgb_tuple(text_color),
                "font_path": font_path,
                "apply_distortion": apply_distortion,
                "random_position": random_position,
                "use_random_bg": use_random_bg,
                "apply_blur": apply_blur,
                "apply_noise": apply_noise,
            }
            save_config("dataset", config_to_save)
            st.success("✅ 설정이 저장되었습니다.")
        except Exception as e:
            st.error(f"설정 저장 중 오류 발생: {str(e)}")

# --- 미리보기 기능 ---
st.markdown("### 1. 설정값으로 예시 이미지 미리보기")
if st.button("🔍 미리보기 샘플 생성"):
    preview_config = load_config("dataset")
    # 색상 값을 튜플로 변환
    preview_config["background"] = ensure_rgb_tuple(
        preview_config.get("background", (255, 255, 255))
    )
    preview_config["text_color"] = ensure_rgb_tuple(
        preview_config.get("text_color", (0, 0, 0))
    )

    sample_text = "".join(preview_config["digits"][: preview_config["length"]])
    try:
        col1, col2 = st.columns(2)
        with col1:
            img_normal, _ = make_image(
                sample_text,
                preview_config,
                font_path=preview_config.get("font_path", "arial.ttf"),
                distorted=False,
            )
            st.image(img_normal, caption="정상 이미지", use_container_width=True)
        with col2:
            img_distorted, _ = make_image(
                sample_text,
                preview_config,
                font_path=preview_config.get("font_path", "arial.ttf"),
                distorted=True,
            )
            st.image(img_distorted, caption="왜곡 이미지", use_container_width=True)
    except Exception as e:
        st.error(f"이미지 생성 중 오류 발생: {e}")

# --- 데이터셋 생성 ---
st.markdown("### 2. 전체 데이터셋 생성 및 다운로드")

if LAST_ZIP_BYTES:
    if LAST_ZIP_META:
        st.info(
            f"📦 생성 시간: {LAST_ZIP_META['time']} | 생성 수량: {LAST_ZIP_META['count']}개 | 소요: {LAST_ZIP_META['duration']}초"
        )

    st.download_button(
        label="📥 마지막 ZIP 다시 다운로드",
        data=LAST_ZIP_BYTES,
        file_name="yolo_dataset.zip",
        mime="application/zip",
    )

if st.button("🚀 데이터셋 생성 및 다운로드"):
    with tempfile.TemporaryDirectory() as tmpdir:
        import time

        start_time = time.time()
        out_dir, count = generate_dataset(load_config("dataset"), tmpdir)
        duration = round(time.time() - start_time, 2)
        # 전역 변수 선언 제거 (이미 모듈 상단에 정의됨)
        LAST_ZIP_BYTES = zip_dataset_folder(out_dir)
        LAST_ZIP_META = {
            "count": count,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": duration,
        }
        st.success(f"총 {count}개의 이미지를 생성하고 압축했습니다.")
        st.download_button(
            label="📥 ZIP 다운로드",
            data=LAST_ZIP_BYTES,
            file_name="yolo_dataset.zip",
            mime="application/zip",
        )
