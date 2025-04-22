# pages/02_모델_학습.py
# 학습 파라미터 저장/불러오기 기능이 추가된 YOLO 모델 학습 페이지

import streamlit as st
import zipfile
import os
from datetime import datetime
from utils.training import run_training
from utils.logger import log_training
from utils.config import load_config, save_config
import matplotlib.pyplot as plt
import pandas as pd
import glob
from PIL import Image

# 페이지 설정
st.set_page_config(page_title="YOLO 모델 학습", page_icon="🧠", layout="wide")

# 페이지 헤더 설정
st.title("🧠 YOLO 모델 학습")

# 현재 설정 불러오기
training_config = load_config("training")  # 학습 설정 불러오기

# 학습 설정이 없으면 기본값으로 초기화
if not training_config:
    training_config = {
        "model_arch": "yolov8n.pt",
        "epochs": 100,
        "batch": 16,
        "imgsz": 640,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "project_name": "ocr_digit",
    }

# 현재 설정된 모델 정보 표시
with st.expander("현재 모델 설정 정보", expanded=False):
    st.info(
        f"""
        **현재 모델 설정:**
        - 모델 경로: {training_config.get('model_path', 'yolov8n.pt')}
        """
    )

# 데이터셋 업로드 섹션
with st.expander("📦 학습용 데이터셋 업로드", expanded=True):
    uploaded_file = st.file_uploader("🔼 ZIP 형식의 학습 데이터셋 업로드", type=["zip"])
    if uploaded_file:
        dataset_dir = "dataset"
        # 기존 데이터셋 삭제
        if os.path.exists(dataset_dir):
            import shutil

            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir, exist_ok=True)

        # 압축 해제
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)

        # 중첩된 dataset 폴더가 있는 경우 자동 이동 처리
        inner_dataset = os.path.join(dataset_dir, "dataset")
        if os.path.exists(inner_dataset):
            for item in os.listdir(inner_dataset):
                src = os.path.join(inner_dataset, item)
                dst = os.path.join(dataset_dir, item)
                shutil.move(src, dst)
            shutil.rmtree(inner_dataset)

        # data.yaml을 최상위로 복사
        for root, dirs, files in os.walk(dataset_dir):
            for f in files:
                if f == "data.yaml":
                    src_path = os.path.join(root, f)
                    dst_path = os.path.join(dataset_dir, "data.yaml")
                    if src_path != dst_path:
                        import shutil

                        shutil.copy(src_path, dst_path)
                    break

        st.session_state.uploaded_filename = uploaded_file.name
        st.success("✅ 압축 해제 완료")

        # 🖼️ 샘플 이미지 미리보기
        with st.expander("🖼️ 샘플 이미지 미리보기", expanded=False):
            image_files = glob.glob(
                os.path.join(dataset_dir, "images", "*.jpg")
            ) + glob.glob(os.path.join(dataset_dir, "images", "*.png"))

            if image_files:
                for i, img_path in enumerate(image_files[:3]):
                    img = Image.open(img_path)
                    st.image(img, caption=os.path.basename(img_path), width=300)
            else:
                st.info("미리볼 이미지가 없습니다. (images/ 폴더를 확인하세요)")

        # 📂 디렉토리 미리보기
        st.subheader("📂 압축 해제된 디렉토리 미리보기")
        with st.expander("📂 디렉토리 보기 (클릭해서 펼치기)", expanded=False):
            for root, dirs, files in os.walk(dataset_dir):
                indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * (
                    root.count(os.sep) - dataset_dir.count(os.sep)
                )
                st.markdown(
                    f"{indent}📂 `{os.path.relpath(root, dataset_dir)}`",
                    unsafe_allow_html=True,
                )
                for f in files:
                    st.markdown(
                        f"{indent}&nbsp;&nbsp;&nbsp;&nbsp;📄 `{f}`",
                        unsafe_allow_html=True,
                    )

# 학습 파라미터 설정 폼
st.subheader("💾 학습 설정")
col1, col2 = st.columns([3, 1])

with col1:
    model_arch = st.selectbox(
        "YOLO 모델",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=(
            [
                "yolov8n.pt",
                "yolov8s.pt",
                "yolov8m.pt",
                "yolov8l.pt",
                "yolov8x.pt",
            ].index(training_config.get("model_arch", "yolov8n.pt"))
            if training_config.get("model_arch")
            in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
            else 0
        ),
        help="YOLO 모델의 크기를 선택합니다.\n\n"
        "📊 모델 크기 비교:\n"
        "- yolov8n.pt: 가장 작고 빠른 모델 (추론 속도 ⚡️⚡️⚡️, 정확도 ★☆☆)\n"
        "- yolov8s.pt: 작은 크기 모델 (추론 속도 ⚡️⚡️, 정확도 ★★☆)\n"
        "- yolov8m.pt: 중간 크기 모델 (추론 속도 ⚡️, 정확도 ★★★)\n"
        "- yolov8l.pt: 큰 크기 모델 (추론 속도 🐢, 정확도 ★★★★)\n"
        "- yolov8x.pt: 가장 큰 모델 (추론 속도 🐢🐢, 정확도 ★★★★★)\n\n"
        "💡 권장: 숫자 인식용으로는 yolov8n.pt나 yolov8s.pt로 시작해보세요.",
    )

epochs = st.slider(
    "Epoch 수",
    1,
    300,
    value=int(training_config.get("epochs", 100)),
    help="전체 데이터셋을 몇 번 반복해서 학습할지 설정합니다.\n\n"
    "📝 epoch 설정 가이드:\n"
    "- 너무 적으면 (< 50): 학습이 충분히 이루어지지 않을 수 있음\n"
    "- 너무 많으면 (> 200): 과적합이 발생할 수 있음\n"
    "- Early Stopping: 성능 개선이 없으면 자동으로 학습 중단\n\n"
    "💡 권장값: 100~150 정도에서 시작하여 조정",
)

batch = st.slider(
    "Batch Size",
    1,
    64,
    value=int(training_config.get("batch", 16)),
    help="한 번에 처리할 이미지 개수입니다.\n\n"
    "🔍 batch size 특징:\n"
    "- 큰 값: 학습이 안정적이지만 더 많은 메모리 필요\n"
    "- 작은 값: 적은 메모리로 가능하나 학습이 불안정할 수 있음\n"
    "- GPU 메모리에 따라 조절 필요\n\n"
    "💡 권장값: 16~32 (GPU 메모리 여유가 있는 경우)",
)

imgsz = st.selectbox(
    "이미지 크기",
    [416, 512, 640],
    index=(
        [416, 512, 640].index(training_config.get("imgsz", 640))
        if training_config.get("imgsz") in [416, 512, 640]
        else 2
    ),
    help="학습에 사용할 이미지 크기(픽셀)를 설정합니다.\n\n"
    "🖼️ 크기별 특징:\n"
    "- 416: 작은 크기, 빠른 학습/추론, 작은 객체 감지력 낮음\n"
    "- 512: 중간 크기, 균형잡힌 성능\n"
    "- 640: 큰 크기, 느린 학습/추론, 작은 객체 감지력 높음\n\n"
    "💡 권장: 숫자 감지의 경우 512나 640 권장",
)

optimizer = st.selectbox(
    "Optimizer",
    ["SGD", "Adam"],
    index=(
        ["SGD", "Adam"].index(training_config.get("optimizer", "Adam"))
        if training_config.get("optimizer") in ["SGD", "Adam"]
        else 1
    ),
    help="학습에 사용할 최적화 알고리즘을 선택합니다.\n\n"
    "🔄 옵티마이저 비교:\n"
    "- SGD (Stochastic Gradient Descent):\n"
    "  • 안정적이지만 수렴이 느림\n"
    "  • 학습률 설정이 중요\n"
    "  • 지역 최적해에 빠질 수 있음\n\n"
    "- Adam (Adaptive Moment Estimation):\n"
    "  • 자동으로 학습률 조절\n"
    "  • 빠른 수렴\n"
    "  • 대부분의 경우 좋은 성능\n\n"
    "💡 권장: 일반적으로 Adam 사용 권장",
)

learning_rate = st.number_input(
    "Learning Rate",
    value=float(training_config.get("learning_rate", 0.001)),
    format="%f",
    min_value=0.0001,
    max_value=0.1,
    step=0.0001,
    help="모델의 학습 속도를 조절하는 학습률입니다.\n\n"
    "📈 학습률 설정 가이드:\n"
    "- 너무 크면 (> 0.01):\n"
    "  • 학습이 불안정해질 수 있음\n"
    "  • 최적점을 건너뛸 수 있음\n\n"
    "- 너무 작으면 (< 0.0001):\n"
    "  • 학습이 매우 느림\n"
    "  • 지역 최적해에 갇힐 수 있음\n\n"
    "💡 권장값:\n"
    "- Adam: 0.001 (기본값)\n"
    "- SGD: 0.01",
)

project_name = st.text_input(
    "프로젝트 이름",
    value=training_config.get("project_name", "ocr_digit"),
    help="학습 결과가 저장될 프로젝트 폴더의 이름입니다.\n\n"
    "📁 저장 위치:\n"
    "- runs/detect/{프로젝트명}/\n"
    "- 같은 이름으로 재실행 시 기존 결과를 덮어씁니다.\n\n"
    "💡 팁: 실험별로 다른 이름을 사용하면 결과를 비교하기 쉽습니다.",
)

# 설정 저장 섹션
st.divider()
st.subheader("⚙️ 설정 저장")
save_col1, save_col2, save_col3 = st.columns([2, 1, 1])

with save_col1:
    save_as_preset = st.text_input("설정 저장 이름", placeholder="새 설정 이름 입력...")

with save_col2:
    if st.button(
        "설정만 저장",
        help="학습을 시작하지 않고 현재 설정만 저장합니다",
        use_container_width=True,
    ):
        current_training = {
            "model_arch": model_arch,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "project_name": project_name,
        }
        save_config("training", current_training)
        st.success("✅ 학습 설정이 저장되었습니다.")

# 학습 실행 섹션
st.divider()
st.subheader("📈 마지막 학습 결과 조회")
if st.button("📂 마지막 학습 결과 보기"):
    st.subheader("📊 학습 결과 요약")
    result_dir = os.path.join(
        "runs", "detect", training_config.get("project_name", "ocr_digit")
    )
    csv_path = os.path.join(result_dir, "results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.line_chart(
            df[["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)"]]
        )
        best_model_path = os.path.join(result_dir, "weights", "best.pt")
        if os.path.exists(best_model_path):
            with open(best_model_path, "rb") as f:
                st.download_button("📥 Best 모델 다운로드", f, file_name="best.pt")
    else:
        st.warning("이전에 저장된 학습 결과가 없습니다.")
    st.subheader("🖼️ 결과 이미지 시각화")
    image_files = [
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "PR_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "labels.jpg",
        "labels_correlogram.jpg",
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg",
        "val_batch0_labels.jpg",
        "val_batch1_labels.jpg",
    ]
    images = []
    for file in image_files:
        img_path = os.path.join(result_dir, file)
        if os.path.exists(img_path):
            images.append((file, img_path))

    if images:
        cols = st.columns(4)
        for i, (name, path) in enumerate(images):
            with open(path, "rb") as f:
                img = f.read()
            cols[i % 4].image(img, caption=name, use_container_width=True)
    else:
        st.info(
            "📁 결과 이미지가 없습니다. runs/detect/{project_name}/ 경로를 확인하세요."
        )


if st.button("🚀 학습 시작"):
    current_training = {
        "model_arch": model_arch,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "project_name": project_name,
    }
    save_config("training", current_training)

    start_time = datetime.now()

    # data.yaml의 상대 경로를 절대 경로로 보정
    import yaml

    yaml_path = os.path.abspath("dataset/data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for key in ["train", "val"]:
            if key in data:
                data[key] = os.path.abspath(data[key]).replace("\\", "/")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

    data_yaml = yaml_path
    yolo_cmd = (
        f"yolo task=detect mode=train model={model_arch} data={data_yaml} "
        f"epochs={epochs} batch={batch} imgsz={imgsz} lr0={learning_rate} "
        f"optimizer={optimizer.lower()} name={project_name}"
    )
    st.info(f"📡 학습 명령 실행 중: {yolo_cmd}")

    with st.spinner("YOLO 학습 진행 중..."):
        result = run_training(yolo_cmd)

    end_time = datetime.now()

    if result.returncode == 0:
        st.success("🎉 학습 성공")
        log_training(start_time, end_time, current_training, "success")

        # 최종 모델 경로를 config.yaml에 저장
        best_model_path = os.path.join(result_dir, "weights", "best.pt")
        if os.path.exists(best_model_path):
            save_config("detection", {"model_path": best_model_path})
            st.info(
                f"✅ 최종 모델 경로가 config.yaml에 저장되었습니다: {best_model_path}"
            )

        # ✅ 학습 결과 시각화 추가
        st.subheader("🖼️ 결과 이미지 시각화")
        image_files = [
            "F1_curve.png",
            "P_curve.png",
            "R_curve.png",
            "PR_curve.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "labels.jpg",
            "labels_correlogram.jpg",
            "val_batch0_pred.jpg",
            "val_batch1_pred.jpg",
            "val_batch0_labels.jpg",
            "val_batch1_labels.jpg",
        ]
        images = []
        for file in image_files:
            img_path = os.path.join(result_dir, file)
            if os.path.exists(img_path):
                images.append((file, img_path))

        if images:
            cols = st.columns(4)
            for i, (name, path) in enumerate(images):
                with open(path, "rb") as f:
                    img = f.read()
                cols[i % 4].image(img, caption=name, use_column_width=True)
        else:
            st.info(
                "📁 결과 이미지가 없습니다. runs/detect/{project_name}/ 경로를 확인하세요."
            )
        st.subheader("📊 학습 결과 요약")
        result_dir = os.path.join("runs", "detect", project_name)
        csv_path = os.path.join(result_dir, "results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.line_chart(
                df[["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)"]]
            )
        else:
            st.warning("results.csv 파일을 찾을 수 없습니다.")

        if os.path.exists(best_model_path):
            with open(best_model_path, "rb") as f:
                st.download_button("📥 Best 모델 다운로드", f, file_name="best.pt")
    else:
        st.error("❌ 학습 실패")
        st.code(result.stdout)
