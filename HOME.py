import streamlit as st
from utils.config import load_config  # 설정 로드 함수

# 페이지 설정
st.set_page_config(page_title="OCR 시스템 홈", page_icon="🏠", layout="wide")

# 페이지 제목
st.title("🏠 OCR 시스템 홈")

# 설정 로드
ui_config = load_config("ui")  # UI 관련 설정 로드

# 자동 리다이렉트 설정 확인
if ui_config.get("auto_redirect_to_detection", False):
    # 감지 페이지로 리다이렉트
    st.experimental_set_query_params(
        page=ui_config.get("detection_page_path", "감지실행")
    )
    st.info("자동으로 감지 페이지로 이동 중입니다...")
else:
    # 기본 홈 화면 메시지
    st.markdown("### 환영합니다! OCR 시스템에 오신 것을 환영합니다.")
    st.markdown("왼쪽 사이드바에서 원하는 페이지를 선택하세요.")
    st.markdown(
        """
    OCR 시스템 주요 기능:
    - 실시간 감지: 카메라를 통해 실시간으로 객체를 감지하고 처리합니다.
    - ROI 설정: 관심 영역(ROI)을 설정하여 특정 영역만 감지할 수 있습니다.
    - 모델 학습: YOLO 모델을 학습시켜 감지 성능을 향상시킬 수 있습니다.
    - 환경 설정: 시스템의 다양한 설정을 사용자 정의할 수 있습니다.
    """
    )
