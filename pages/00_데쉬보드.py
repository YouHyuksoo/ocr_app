# pages/01_대시보드.py
# OCR 시스템 대시보드 요약 페이지

# 필요한 라이브러리 불러오기
import streamlit as st  # 웹 기반 대시보드 생성을 위한 라이브러리
import pandas as pd  # 데이터 분석 및 조작을 위한 라이브러리
from datetime import datetime  # 날짜 및 시간 관련 기능 제공
from utils.config import load_config  # 설정 파일 로드 기능
from pathlib import Path  # 파일 경로 처리를 위한 라이브러리

# 페이지 설정 (각 페이지마다 필요)
st.set_page_config(page_title="OCR 대시보드 요약", page_icon="📊", layout="wide")

# 페이지 메인 제목 설정
st.title("📊 OCR 실시간 대시보드 요약")

# 설정 파일에서 모델 및 ROI 정보 로드
roi_config = load_config("roi")  # ROI 설정 로드
camera_config = load_config("camera")  # 카메라 설정 로드
training_config = load_config("training")  # 학습 설정 로드
model_path = training_config.get("model_path", "yolov8n.pt")  # 모델 경로

# ROI(관심 영역) 설정값 불러오기
x, y, width, height = (
    roi_config.get("x", 100),  # ROI 좌측 상단의 x 좌표 (기본값: 100)
    roi_config.get("y", 100),  # ROI 좌측 상단의 y 좌표 (기본값: 100)
    roi_config.get("width", 200),  # ROI 너비 (기본값: 200)
    roi_config.get("height", 100),  # ROI 높이 (기본값: 100)
)

# 설정 파일에서 카메라 해상도 정보 로드
camera_width = camera_config.get("width", 320)  # 기본값: 320
camera_height = camera_config.get("height", 240)  # 기본값: 240

# 화면을 2개 열로 분할하여 정보 표시
col1, col2 = st.columns(2)  # 동일한 너비의 두 열 생성

# 첫 번째 열: 모델 상태 정보 표시
with col1:
    st.subheader("📦 모델 상태")
    if Path(model_path).exists():
        st.success("✅ 모델 로드됨")
        st.markdown(f"**사용 모델:** `{model_path}`")
        size_mb = round(Path(model_path).stat().st_size / 1024 / 1024, 2)
        st.markdown(f"**파일 크기:** {size_mb} MB")
        model_dir = Path(model_path).parent
        st.markdown(f"**모델 위치 디렉토리:** `{model_dir}`")
    else:
        st.error(f"❌ 모델 파일이 존재하지 않습니다: {model_path}")

# 두 번째 열: ROI 및 카메라 설정 정보 표시
with col2:
    st.subheader("🔲 ROI 및 카메라 설정 상태")  # 섹션 제목
    st.markdown(f"**X, Y:** ({x}, {y})")  # ROI 시작 좌표 표시
    st.markdown(f"**Width × Height (ROI):** {width} × {height}")  # ROI 크기 표시
    st.markdown(
        f"**Width × Height (카메라):** {camera_width} × {camera_height}"
    )  # 카메라 해상도 표시

# 구분선 추가
st.divider()  # 시각적 구분을 위한 수평선

# 로그 요약 섹션
log_path = "logs/detections.csv"  # 감지 로그 파일 경로
plc_log_path = "logs/plc_sent.csv"  # PLC 전송 로그 파일 경로
st.subheader("🧾 최근 감지 및 PLC 로그 요약")  # 섹션 제목

# 로그 정보를 2개 열로 분할하여 표시
cols = st.columns(2)  # 동일한 너비의 두 열 생성

# 첫 번째 열: 감지 로그 표시
with cols[0]:
    st.markdown("#### 🔢 감지 로그")  # 소제목
    if Path(log_path).exists():  # 감지 로그 파일이 존재하는 경우
        df_log = pd.read_csv(log_path)  # CSV 파일을 판다스 데이터프레임으로 읽기
        st.markdown(f"- 총 감지 수: **{len(df_log)}건**")  # 총 감지 수 표시
        if not df_log.empty:  # 로그 데이터가 비어있지 않은 경우
            st.dataframe(
                df_log.tail(5), use_container_width=True
            )  # 최근 5개 로그 항목 테이블로 표시
        # 로그 삭제 버튼 추가
        if st.button("🔄 감지 로그 삭제"):
            Path(log_path).unlink()  # 로그 파일 삭제
            st.warning("감지 로그가 삭제되었습니다.")  # 경고 메시지 표시
    else:  # 감지 로그 파일이 존재하지 않는 경우
        st.info("감지 로그 파일 없음")  # 정보 메시지 표시

# 두 번째 열: PLC 전송 로그 표시
with cols[1]:
    st.markdown("#### 🔌 PLC 전송 로그")  # 소제목
    if Path(plc_log_path).exists():  # PLC 로그 파일이 존재하는 경우
        # CSV 파일을 판다스 데이터프레임으로 읽기 (열 이름 지정)
        df_plc = pd.read_csv(plc_log_path, names=["timestamp", "number", "status"])
        st.markdown(f"- 총 전송 수: **{len(df_plc)}건**")  # 총 전송 수 표시
        if not df_plc.empty:  # 로그 데이터가 비어있지 않은 경우
            st.dataframe(
                df_plc.tail(5), use_container_width=True
            )  # 최근 5개 로그 항목 테이블로 표시
        # 로그 삭제 버튼 추가
        if st.button("🔄 PLC 로그 삭제"):
            Path(plc_log_path).unlink()  # 로그 파일 삭제
            st.warning("PLC 로그가 삭제되었습니다.")  # 경고 메시지 표시
    else:  # PLC 로그 파일이 존재하지 않는 경우
        st.info("PLC 로그 파일 없음")  # 정보 메시지 표시

# 구분선 추가
st.divider()  # 시각적 구분을 위한 수평선

# 추가 대시보드 기능 - 시간별 감지 추이 (데이터가 있는 경우)
if (
    Path(log_path).exists() and Path(log_path).stat().st_size > 0
):  # 로그 파일이 존재하고 비어있지 않은 경우
    try:
        st.subheader("📈 시간별 감지 추이")  # 섹션 제목
        df_log = pd.read_csv(log_path)  # 로그 파일 읽기

        # 타임스탬프 컬럼이 있다고 가정하고 처리
        if "timestamp" in df_log.columns:  # timestamp 열이 있는 경우
            df_log["timestamp"] = pd.to_datetime(
                df_log["timestamp"]
            )  # 타임스탬프 문자열을 datetime 객체로 변환
            df_log["hour"] = df_log["timestamp"].dt.hour  # 시간 정보 추출

            # 시간별 그룹화하여 감지 횟수 계산
            hourly_counts = df_log.groupby("hour").size().reset_index(name="count")

            # 차트 표시 (막대 그래프)
            st.bar_chart(hourly_counts.set_index("hour"))  # 시간별 감지 횟수 그래프

            # 오늘의 활동 요약
            # 현재 날짜와 같은 데이터만 필터링하여 오늘의 감지 수 계산
            st.markdown(
                f"**오늘의 총 감지 수:** {len(df_log[df_log['timestamp'].dt.date == datetime.now().date()])}건"
            )
    except Exception as e:  # 오류 발생 시
        st.warning(f"데이터 분석 중 오류 발생: {e}")  # 경고 메시지 표시

# 푸터 - 마지막 업데이트 시간 표시
st.markdown("---")  # 구분선
st.markdown(
    f"*마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
)  # 현재 시간을 마지막 업데이트 시간으로 표시
