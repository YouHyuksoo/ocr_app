# pages/05_환경설정.py
# OCR 시스템 통합 환경설정 페이지 - 모든 설정을 config.toml에 저장

import streamlit as st
from pathlib import Path
from utils.config import load_config, save_config

# 페이지 설정
st.set_page_config(page_title="OCR 시스템 환경설정", page_icon="⚙️", layout="wide")

st.title("⚙️ OCR 시스템 환경설정")
st.markdown("---")

# 설정 탭 생성
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["시스템 설정", "ROI 설정", "감지 설정", "PLC 설정", "카메라 설정"]
)

# 탭 1: 시스템 설정
with tab1:
    st.subheader("일반 설정")

    # OCR 시스템 설정 로드
    ocr_config = load_config("ocr_system")
    ui_config = load_config("ui")
    theme_config = load_config("theme")
    server_config = load_config("server")

    # 자동 시작 설정
    auto_redirect = st.checkbox(
        "홈페이지에서 자동으로 감지 페이지로 이동",
        value=ui_config.get("auto_redirect_to_detection", False),
        help="활성화하면 홈페이지 접속 시 자동으로 감지 페이지로 이동합니다.",
    )

    auto_start = st.checkbox(
        "감지 페이지 진입 시 자동으로 감지 시작",
        value=ocr_config.get("auto_start_detection", False),
        help="활성화하면 감지 페이지 로드 시 자동으로 객체 감지를 시작합니다.",
    )

    detection_page = st.text_input(
        "감지 페이지 경로",
        value=ui_config.get("detection_page_path", "감지실행"),
        help="멀티페이지 앱에서 감지 페이지의 경로명 (숫자 접두사 제외, 예: '01_감지실행' → '감지실행')",
    )

    # 테마 설정
    st.subheader("테마 설정")
    primary_color = st.color_picker(
        "주 색상", value=theme_config.get("primaryColor", "#ff4b4b")
    )

    # 고급 설정
    st.subheader("고급 설정")
    with st.expander("서버 설정"):
        headless = st.checkbox(
            "Headless 모드",
            value=server_config.get("headless", True),
            help="브라우저 창을 자동으로 열지 않습니다.",
        )

        run_on_save = st.checkbox(
            "저장 시 자동 재실행",
            value=server_config.get("runOnSave", True),
            help="코드 변경 시 자동으로 서버를 재시작합니다.",
        )

    # 시스템 설정 저장 버튼
    if st.button("시스템 설정 저장", key="save_system"):
        # 설정 업데이트
        save_config("ocr_system", {"auto_start_detection": auto_start})
        save_config(
            "ui",
            {
                "auto_redirect_to_detection": auto_redirect,
                "detection_page_path": detection_page,
            },
        )
        save_config("theme", {"primaryColor": primary_color})
        save_config("server", {"headless": headless, "runOnSave": run_on_save})

        st.success("✅ 시스템 설정이 성공적으로 저장되었습니다.")
        st.info("변경된 설정을 적용하려면 페이지를 새로고침하세요.")

# 탭 2: ROI 설정
with tab2:
    st.subheader("관심 영역(ROI) 설정")

    # 설정 파일에서 ROI 설정 로드
    roi_config = load_config("roi") or {"x": 100, "y": 100, "width": 200, "height": 100}

    # ROI 설정
    x = st.number_input("X 좌표", min_value=0, value=roi_config.get("x", 100))
    y = st.number_input("Y 좌표", min_value=0, value=roi_config.get("y", 100))
    width = st.number_input("너비", min_value=10, value=roi_config.get("width", 200))
    height = st.number_input("높이", min_value=10, value=roi_config.get("height", 100))

    # ROI 설정 저장 버튼
    if st.button("ROI 설정 저장", key="save_roi"):
        save_config("roi", {"x": x, "y": y, "width": width, "height": height})
        st.success("✅ ROI 설정이 성공적으로 저장되었습니다.")
        st.info("변경된 설정을 적용하려면 페이지를 새로고침하세요.")

# 탭 3: 감지 설정
with tab3:
    st.subheader("감지 설정")

    # 설정 파일에서 감지 설정 로드
    detection_config = load_config("detection") or {
        "model_path": "yolov8n.pt",
        "conf": 0.25,
        "iou": 0.45,
        "agnostic_nms": False,
        "max_det": 10,
        "entry_direction": "top_to_bottom",  # 기본값 추가
        "digit_count": 3,  # 기본값 추가
    }

    # 모델 설정
    model_path = st.text_input(
        "모델 파일 경로",
        value=detection_config.get("model_path", "yolov8n.pt"),
        help="YOLO 모델 파일의 경로",
    )

    # 진입 방향 설정
    entry_direction = st.selectbox(
        "객체 진입 방향",
        options=["top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left"],
        index=[
            "top_to_bottom",
            "bottom_to_top",
            "left_to_right",
            "right_to_left",
        ].index(detection_config.get("entry_direction", "top_to_bottom")),
        help="객체가 진입하는 방향을 설정합니다",
        format_func=lambda x: {
            "top_to_bottom": "위에서 아래로",
            "bottom_to_top": "아래에서 위로",
            "left_to_right": "왼쪽에서 오른쪽으로",
            "right_to_left": "오른쪽에서 왼쪽으로",
        }[x],
    )

    # 숫자 자리수 설정
    digit_count = st.number_input(
        "감지할 숫자 자리수",
        min_value=1,
        max_value=10,
        value=detection_config.get("digit_count", 3),
        help="감지할 숫자의 자리수를 설정합니다",
    )

    # 기존 감지 설정들...
    conf = st.slider(
        "Confidence Threshold",
        0.1,
        1.0,
        value=detection_config.get("conf", 0.25),
        step=0.05,
        help="객체 감지의 확신도를 조절하는 임계값입니다. (0.1~1.0)\n\n"
        "- 값이 높을수록(1에 가까울수록) 확실한 경우에만 감지\n"
        "- 값이 낮을수록(0.1에 가까울수록) 불확실해도 감지\n"
        "- 너무 높으면 실제 객체를 놓칠 수 있음\n"
        "- 너무 낮으면 잘못된 객체를 감지할 수 있음\n\n"
        "💡 권장값: 0.25~0.45",
    )

    iou = st.slider(
        "IoU Threshold",
        0.1,
        1.0,
        value=detection_config.get("iou", 0.45),
        step=0.05,
        help="겹치는 객체를 처리하는 IoU(Intersection over Union) 임계값입니다. (0.1~1.0)\n\n"
        "- 값이 높을수록(1에 가까울수록) 겹치는 영역이 많아도 다른 객체로 인식\n"
        "- 값이 낮을수록(0.1에 가까울수록) 겹치는 영역이 있으면 하나의 객체로 통합\n"
        "- 너무 높으면 하나의 객체가 여러 개로 중복 감지될 수 있음\n"
        "- 너무 낮으면 서로 다른 객체가 하나로 합쳐질 수 있음\n\n"
        "💡 권장값: 0.45~0.65",
    )

    agnostic_nms = st.checkbox(
        "Agnostic NMS",
        value=detection_config.get("agnostic_nms", False),
        help="클래스 무관 비최대 억제(Non-Maximum Suppression) 설정입니다.\n\n"
        "- 활성화: 서로 다른 클래스의 객체라도 겹치는 영역이 많으면 하나로 처리\n"
        "- 비활성화: 서로 다른 클래스의 객체는 겹쳐도 별도로 처리\n\n"
        "📝 사용 예시:\n"
        "- 숫자 '8'과 '3'이 겹쳐 있을 때\n"
        "  - 활성화: IoU가 높으면 신뢰도가 더 높은 하나만 선택\n"
        "  - 비활성화: 다른 숫자로 인식되면 둘 다 검출\n\n"
        "💡 권장: 숫자가 겹치지 않는 경우 비활성화",
    )
    max_det = st.number_input(
        "Max Detections",
        min_value=1,
        max_value=100,
        value=detection_config.get("max_det", 10),
        help="한 프레임에서 감지할 최대 객체 수를 설정합니다. 값이 클수록 더 많은 객체를 감지하지만 처리 시간이 늘어날 수 있습니다.",
    )

    # 감지 설정 저장 버튼
    if st.button("감지 설정 저장", key="save_detection"):
        save_config(
            "detection",
            {
                "model_path": model_path,
                "conf": conf,
                "iou": iou,
                "agnostic_nms": agnostic_nms,
                "max_det": max_det,
                "entry_direction": entry_direction,
                "digit_count": digit_count,
            },
        )
        st.success("✅ 감지 설정이 성공적으로 저장되었습니다.")
        st.info("변경된 설정을 적용하려면 페이지를 새로고침하세요.")

# 탭 4: PLC 설정
with tab4:
    st.subheader("📡 PLC 통신 설정")

    # PLC 설정 로드
    plc_config = load_config("plc_defaults")

    plc_enabled = st.checkbox(
        "PLC 통신 활성화",
        value=plc_config.get("enabled", True),
        help="PLC 통신 기능의 기본 활성화 여부를 설정합니다.",
    )

    plc_ip = st.text_input(
        "기본 PLC IP 주소",
        value=plc_config.get("ip", "192.168.0.10"),
        help="PLC 장치의 IP 주소를 입력하세요.",
    )

    plc_port = st.number_input(
        "기본 통신 포트",
        min_value=1,
        max_value=65535,
        value=plc_config.get("port", 502),
        help="PLC 통신 포트 (기본값: 502 - Modbus 표준 포트)",
    )

    plc_register = st.number_input(
        "기본 레지스터 주소",
        min_value=0,
        value=plc_config.get("register", 100),
        help="데이터를 쓸 PLC 레지스터 시작 주소",
    )

    plc_retry = st.number_input(
        "기본 재시도 횟수",
        min_value=0,
        max_value=10,
        value=plc_config.get("retry", 2),
        help="통신 실패 시 재시도 횟수 (0-10)",
    )

    # PLC 설정 저장 버튼
    if st.button("PLC 설정 저장", key="save_plc"):
        save_config(
            "plc_defaults",
            {
                "enabled": plc_enabled,
                "ip": plc_ip,
                "port": plc_port,
                "register": plc_register,
                "retry": plc_retry,
            },
        )

        st.success("✅ PLC 설정이 성공적으로 저장되었습니다.")
        st.info("변경된 설정을 적용하려면 페이지를 새로고침하세요.")

# 탭 5: 카메라 설정
with tab5:
    st.subheader("📷 카메라 설정")

    # 카메라 설정 로드
    camera_config = load_config("camera")

    camera_index = st.number_input(
        "카메라 소스 인덱스",
        min_value=0,
        max_value=10,
        value=camera_config.get("index", 0),
        help="사용할 카메라의 인덱스를 설정합니다. (기본: 0-내장 카메라, 1,2...-외장 카메라)",
    )

    camera_width = st.number_input(
        "카메라 너비 (픽셀)",
        min_value=320,
        max_value=3840,
        value=camera_config.get("width", 1024),
        step=10,
        help="카메라의 해상도 너비를 픽셀 단위로 설정합니다.",
    )

    camera_height = st.number_input(
        "카메라 높이 (픽셀)",
        min_value=240,
        max_value=2160,
        value=camera_config.get("height", 768),
        step=10,
        help="카메라의 해상도 높이를 픽셀 단위로 설정합니다.",
    )

    # 카메라 설정 저장 버튼
    if st.button("카메라 설정 저장", key="save_camera"):
        save_config(
            "camera",
            {
                "index": camera_index,  # 카메라 인덱스 추가
                "width": camera_width,
                "height": camera_height,
            },
        )

        st.success("✅ 카메라 설정이 성공적으로 저장되었습니다.")
        st.info("변경된 설정을 적용하려면 페이지를 새로고침하세요.")

# 푸터
st.markdown("---")
st.info(
    """
### 설정 파일 정보
모든 설정은 `.streamlit/config.toml` 파일에 통합 저장됩니다.
"""
)

# 설정 파일 내용 보기 옵션
with st.expander("설정 파일 전체 내용 보기"):
    config_path = Path(".streamlit/config.toml")  # 설정 파일 경로
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config_content = f.read()
        st.code(config_content, language="toml")
    else:
        st.warning("설정 파일이 아직 생성되지 않았습니다.")
