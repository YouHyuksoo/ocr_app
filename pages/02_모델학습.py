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
import yaml

# 학습 결과 저장 경로 설정
RESULTS_DIR = "runs/detect"  # 기본 결과 저장 디렉토리
os.makedirs(RESULTS_DIR, exist_ok=True)

# 페이지 설정
st.set_page_config(page_title="YOLO 모델 학습", page_icon="🧠", layout="wide")

# 사이드바에 설정 저장 섹션 추가
with st.sidebar:
    st.subheader("⚙️ 학습 설정 저장")
    save_as_preset = st.text_input(
        "설정 이름",
        placeholder="저장할 설정 이름 입력...",
        help="현재 설정을 저장할 이름을 입력하세요",
    )

    if st.button("💾 현재 설정 저장", use_container_width=True):
        if save_as_preset.strip():  # 설정 이름이 비어있지 않은 경우에만 저장
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
        else:
            st.warning("⚠️ 설정 이름을 입력해주세요.")

    st.divider()

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
st.subheader("📦 학습용 데이터셋")


# data.yaml의 경로 구조 검증 및 수정
def validate_dataset_structure(dataset_dir):
    """데이터셋 디렉토리 구조를 검증하고 필요한 경우 수정합니다."""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        st.error("❌ data.yaml 파일을 찾을 수 없습니다.")
        return False

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 필수 키 확인
    required_keys = ["train", "val", "nc", "names"]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        st.error(f"❌ data.yaml에 필수 키가 없습니다: {', '.join(missing_keys)}")
        return False

    # 이미지 디렉토리 구조 확인
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    for dir_type in ["train", "val"]:
        img_dir = os.path.join(images_dir, dir_type)
        label_dir = os.path.join(labels_dir, dir_type)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir, exist_ok=True)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir, exist_ok=True)

        # data.yaml의 경로 업데이트
        data[dir_type] = os.path.join(images_dir, dir_type)

    # data.yaml 업데이트
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    return True


def validate_dataset_for_training(dataset_dir):
    """
    학습에 필요한 데이터셋 구조와 데이터를 검증합니다.
    """
    validation_results = {
        "is_valid": True,
        "messages": [],
        "stats": {
            "train": {"images": 0, "labels": 0},
            "val": {"images": 0, "labels": 0},
        },
    }

    # 1. data.yaml 파일 검증
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        validation_results["is_valid"] = False
        validation_results["messages"].append("❌ data.yaml 파일이 없습니다.")
        return validation_results

    # 2. yaml 파일 내용 검증
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            required_keys = ["train", "val", "nc", "names"]
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                validation_results["is_valid"] = False
                validation_results["messages"].append(
                    f"❌ data.yaml에 필수 키가 없습니다: {', '.join(missing_keys)}"
                )
    except Exception as e:
        validation_results["is_valid"] = False
        validation_results["messages"].append(f"❌ data.yaml 파일 읽기 오류: {str(e)}")
        return validation_results

    # 3. 디렉토리 구조 확인
    for split in ["train", "val"]:
        img_dir = os.path.join(dataset_dir, "images", split)
        label_dir = os.path.join(dataset_dir, "labels", split)

        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            validation_results["is_valid"] = False
            validation_results["messages"].append(
                f"❌ {split} 폴더 구조가 올바르지 않습니다."
            )
            continue

        # 4. 이미지와 라벨 파일 수 확인
        images = (
            glob.glob(os.path.join(img_dir, "*.jpg"))
            + glob.glob(os.path.join(img_dir, "*.jpeg"))
            + glob.glob(os.path.join(img_dir, "*.png"))
        )
        labels = glob.glob(os.path.join(label_dir, "*.txt"))

        validation_results["stats"][split]["images"] = len(images)
        validation_results["stats"][split]["labels"] = len(labels)

        if len(images) == 0:
            validation_results["is_valid"] = False
            validation_results["messages"].append(f"❌ {split} 이미지가 없습니다.")

        if len(labels) == 0:
            validation_results["is_valid"] = False
            validation_results["messages"].append(f"❌ {split} 라벨이 없습니다.")

        if len(images) != len(labels):
            validation_results["messages"].append(
                f"⚠️ {split} 이미지({len(images)})와 라벨({len(labels)}) 수가 다릅니다!"
            )

    # 5. 검증 데이터 비율 확인
    total_images = (
        validation_results["stats"]["train"]["images"]
        + validation_results["stats"]["val"]["images"]
    )

    if total_images > 0:
        val_ratio = validation_results["stats"]["val"]["images"] / total_images
        if val_ratio < 0.1:  # 검증 데이터가 10% 미만인 경우
            validation_results["messages"].append(
                f"⚠️ 검증 데이터 비율이 너무 낮습니다 ({val_ratio:.1%}). 10% 이상을 권장합니다."
            )

    return validation_results


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

    # 데이터셋 구조 검증
    if not validate_dataset_structure(dataset_dir):
        st.error("❌ 데이터셋 구조가 올바르지 않습니다.")
        st.info(
            """
        필요한 디렉토리 구조:
        dataset/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
        """
        )
        st.stop()

    st.session_state.uploaded_filename = uploaded_file.name
    st.success("✅ 압축 해제 완료")

    # 🖼️ 샘플 이미지 미리보기
    st.subheader("🖼️ 샘플 이미지 미리보기")
    image_files = glob.glob(os.path.join(dataset_dir, "images", "*.jpg")) + glob.glob(
        os.path.join(dataset_dir, "images", "*.png")
    )

    if image_files:
        cols = st.columns(3)
        for i, img_path in enumerate(image_files[:3]):
            img = Image.open(img_path)
            cols[i].image(img, caption=os.path.basename(img_path), width=300)
    else:
        st.info("미리볼 이미지가 없습니다. (images/ 폴더를 확인하세요)")

    # 📂 디렉토리 미리보기
    st.subheader("📂 데이터셋 구조")
    for root, dirs, files in os.walk(dataset_dir):
        indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * (
            root.count(os.sep) - dataset_dir.count(os.sep)
        )
        st.markdown(
            f"{indent}📂 `{os.path.relpath(root, dataset_dir)}`", unsafe_allow_html=True
        )
        for f in files:
            st.markdown(
                f"{indent}&nbsp;&nbsp;&nbsp;&nbsp;📄 `{f}`", unsafe_allow_html=True
            )

# 학습 파라미터 설정 폼
st.subheader("💾 학습 설정")

# 모델 선택
model_arch = st.selectbox(
    "YOLO 모델",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
    index=(
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"].index(
            training_config.get("model_arch", "yolov8n.pt")
        )
        if training_config.get("model_arch")
        in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        else 0
    ),
    help="YOLO 모델의 크기를 선택합니다...",  # 기존 help 텍스트 유지
)

# Epochs와 Batch Size를 한 줄에
col1, col2 = st.columns(2)
with col1:
    epochs = st.slider(
        "Epoch 수",
        1,
        300,
        value=int(training_config.get("epochs", 100)),
        help="전체 데이터셋을 몇 번 반복해서 학습할지 설정합니다...",  # 기존 help 텍스트 유지
    )
with col2:
    batch = st.slider(
        "Batch Size",
        1,
        64,
        value=int(training_config.get("batch", 16)),
        help="한 번에 처리할 이미지 개수입니다...",  # 기존 help 텍스트 유지
    )

# 이미지 크기, Optimizer, Learning Rate를 한 줄에
col3, col4, col5 = st.columns(3)
with col3:
    imgsz = st.selectbox(
        "이미지 크기",
        [416, 512, 640],
        index=(
            [416, 512, 640].index(training_config.get("imgsz", 640))
            if training_config.get("imgsz") in [416, 512, 640]
            else 2
        ),
        help="학습에 사용할 이미지 크기(픽셀)를 설정합니다...",  # 기존 help 텍스트 유지
    )
with col4:
    optimizer = st.selectbox(
        "Optimizer",
        ["SGD", "Adam"],
        index=(
            ["SGD", "Adam"].index(training_config.get("optimizer", "Adam"))
            if training_config.get("optimizer") in ["SGD", "Adam"]
            else 1
        ),
        help="학습에 사용할 최적화 알고리즘을 선택합니다...",  # 기존 help 텍스트 유지
    )
with col5:
    learning_rate = st.number_input(
        "Learning Rate",
        value=float(training_config.get("learning_rate", 0.001)),
        format="%f",
        min_value=0.0001,
        max_value=0.1,
        step=0.0001,
        help="모델의 학습 속도를 조절하는 학습률입니다...",  # 기존 help 텍스트 유지
    )

# 디바이스 선택 옵션 추가
device = st.selectbox(
    "학습 디바이스",
    options=["cpu", "cuda"],
    index=0,
    help="학습에 사용할 디바이스를 선택합니다.\n\n"
    "- CPU: 모든 환경에서 사용 가능하나 학습 속도가 느림\n"
    "- CUDA: NVIDIA GPU가 있는 경우 선택. 빠른 학습 가능",
)

# 프로젝트 이름 설정
project_name = st.text_input(
    "프로젝트 이름",
    value=training_config.get("project_name", "ocr_digit"),
    help="학습 결과가 저장될 프로젝트 폴더의 이름입니다...",  # 기존 help 텍스트 유지
)

# 학습 시작 버튼
if st.button("🚀 학습 시작", use_container_width=True):
    try:
        # 데이터셋 검증
        validation_results = validate_dataset_for_training("dataset")

        if not validation_results["is_valid"]:
            st.error("❌ 데이터셋 검증 실패")
            for msg in validation_results["messages"]:
                st.error(msg)
            st.stop()

        # YOLO 명령어 생성
        data_yaml = os.path.abspath("dataset/data.yaml")
        yolo_cmd = (
            f"yolo task=detect mode=train model={model_arch} data={data_yaml} "
            f"epochs={epochs} batch={batch} imgsz={imgsz} lr0={learning_rate} "
            f"optimizer={optimizer.lower()} project={RESULTS_DIR} name={project_name} "
            f"device={device} verbose=True"  # device 옵션 수정
        )

        # 학습 실행
        with st.spinner("🚀 학습 진행 중..."):
            start_time = datetime.now()
            st.info(
                "학습이 시작되었습니다. CPU 모드로 실행됩니다. 터미널에서 진행 상황을 확인하세요."
            )

            # YOLO 학습 실행
            result = run_training(yolo_cmd)

            # 표준 출력 처리 개선
            if result.stdout:
                if hasattr(result.stdout, "decode"):
                    print(result.stdout.decode("utf-8", errors="ignore"))
                else:
                    print(result.stdout)

            # 오류 출력 확인
            if result.stderr:
                if hasattr(result.stderr, "decode"):
                    error_msg = result.stderr.decode("utf-8", errors="ignore")
                    print("Error output:", error_msg)
                else:
                    print("Error output:", result.stderr)

            end_time = datetime.now()
            duration = end_time - start_time

            if result.returncode == 0:
                # 학습 결과 저장
                save_path = os.path.join(
                    RESULTS_DIR, project_name, "weights", "best.pt"
                )
                if os.path.exists(save_path):
                    # detection 설정 업데이트
                    detection_config = load_config("detection") or {}
                    detection_config["model_path"] = save_path
                    save_config("detection", detection_config)

                    st.success(
                        f"""✅ 학습이 완료되었습니다!
                    - 소요 시간: {duration}
                    - 모델 저장 위치: {save_path}"""
                    )
                else:
                    st.warning("⚠️ 학습은 완료되었으나 모델 파일을 찾을 수 없습니다.")
            else:
                st.error("❌ 학습 중 오류가 발생했습니다.")
                if result.stderr:
                    st.code(result.stderr.decode(), language="bash")

    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
