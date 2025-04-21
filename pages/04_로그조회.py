import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from pathlib import Path

st.header("📄 감지 로그 및 시각화")

# 감지 로그 보기
log_path = "logs/detections.csv"
if Path(log_path).exists():
    try:
        df_log = pd.read_csv(log_path)
        search_date = st.date_input("📅 날짜 필터", value=datetime.today())
        filtered_df = df_log[
            df_log["timestamp"].str.startswith(search_date.strftime("%Y-%m-%d"))
        ]

        selected_digit = st.selectbox(
            "🔢 특정 숫자 필터링",
            options=["전체"]
            + sorted(set(",".join(df_log["digits"].dropna()).split(","))),
        )
        if selected_digit != "전체":
            filtered_df = filtered_df[
                filtered_df["digits"].str.contains(selected_digit)
            ]

        st.dataframe(filtered_df, use_container_width=True)
        st.download_button(
            "📥 로그 CSV 다운로드",
            filtered_df.to_csv(index=False),
            file_name="filtered_log.csv",
        )
    except Exception as e:
        st.error(f"❌ 감지 로그 로딩 실패: {e}")
else:
    st.info("감지 로그 파일이 존재하지 않습니다.")

# 감지 이미지 갤러리
st.subheader("🖼️ 감지 이미지 갤러리")
snapshot_dir = Path("logs/snapshots")
if snapshot_dir.exists():
    image_files = sorted(snapshot_dir.glob("*.jpg"), reverse=True)
    cols = st.columns(3)
    for idx, img_path in enumerate(image_files[:9]):
        with cols[idx % 3]:
            st.image(str(img_path), caption=img_path.name, use_column_width=True)
            with open(img_path, "rb") as f:
                st.download_button(
                    "📥 다운로드", f, file_name=img_path.name, key=img_path.name
                )
else:
    st.info("감지된 이미지가 없습니다.")

# PLC 전송 로그 보기
st.subheader("📋 PLC 전송 로그")
plc_log_path = "logs/plc_sent.csv"
if Path(plc_log_path).exists():
    try:
        df_plc = pd.read_csv(plc_log_path, names=["timestamp", "number", "status"])
        search_date_plc = st.date_input(
            "PLC 날짜 필터", value=datetime.today(), key="plc_date"
        )
        filtered_plc = df_plc[
            df_plc["timestamp"].str.startswith(search_date_plc.strftime("%Y-%m-%d"))
        ]
        st.dataframe(filtered_plc, use_container_width=True)
        st.download_button(
            "📥 PLC 로그 다운로드",
            filtered_plc.to_csv(index=False),
            file_name="plc_log.csv",
        )
    except Exception as e:
        st.error(f"❌ PLC 로그 로딩 실패: {e}")
else:
    st.info("PLC 전송 로그 파일이 없습니다.")

# 숫자 빈도수 시각화
st.subheader("📊 숫자 인식 빈도 그래프")
if Path(log_path).exists():
    try:
        df_log = pd.read_csv(log_path)
        all_digits = ",".join(df_log["digits"].dropna()).split(",")
        counter = Counter(all_digits)
        df_plot = pd.DataFrame(counter.items(), columns=["숫자", "빈도수"]).sort_values(
            "숫자"
        )
        fig, ax = plt.subplots()
        ax.bar(df_plot["숫자"], df_plot["빈도수"])
        ax.set_xlabel("숫자")
        ax.set_ylabel("빈도수")
        ax.set_title("인식된 숫자 빈도")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"📉 빈도 그래프 표시 실패: {e}")
else:
    st.info("아직 로그가 없어 시각화를 표시할 수 없습니다.")
