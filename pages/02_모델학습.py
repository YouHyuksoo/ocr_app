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

# 설정에서 경로 로드
paths_config = load_config("paths")
RESULTS_DIR = paths_config.get("results_dir", "runs/detect")  # 기본값 설정
os.makedirs(RESULTS_DIR, exist_ok=True)

# 페이지 설정
st.set_page_config(page_title="YOLO 모델 학습", page_icon="🧠", layout="wide")

# 설정 불러오기 (한 번만 실행)
if "training_config" not in st.session_state:
    st.session_state.training_config = load_config("training")

# 사이드바에 설정 관리 섹션
with st.sidebar:
    st.subheader("⚙️ 학습 설정 관리")

    # 현재 설정 저장하기
    if st.button("💾 현재 설정 저장", use_container_width=True):
        current_config = {
            "model_arch": st.session_state.training_config.get(
                "model_arch", "yolov8n.pt"
            ),
            "epochs": st.session_state.training_config.get("epochs", 100),
            "batch": st.session_state.training_config.get("batch", 16),
            "imgsz": st.session_state.training_config.get("imgsz", 640),
            "optimizer": st.session_state.training_config.get("optimizer", "Adam"),
            "learning_rate": st.session_state.training_config.get(
                "learning_rate", 0.001
            ),
            "project_name": st.session_state.training_config.get(
                "project_name", "ocr_digit"
            ),
            "device": st.session_state.training_config.get("device", "cpu"),
        }
        # 현재 설정을 저장
        save_config("training", current_config)
        st.success("✅ 현재 설정이 저장되었습니다.")

# 페이지 제목
st.title("🧠 YOLO 모델 학습")

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
            st.session_state.training_config.get("model_arch", "yolov8n.pt")
        )
        if st.session_state.training_config.get("model_arch")
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
        value=int(st.session_state.training_config.get("epochs", 100)),
        help="전체 데이터셋을 몇 번 반복해서 학습할지 설정합니다...",  # 기존 help 텍스트 유지
    )
with col2:
    batch = st.slider(
        "Batch Size",
        1,
        64,
        value=int(st.session_state.training_config.get("batch", 16)),
        help="한 번에 처리할 이미지 개수입니다...",  # 기존 help 텍스트 유지
    )

# 이미지 크기, Optimizer, Learning Rate를 한 줄에
col3, col4, col5 = st.columns(3)
with col3:
    imgsz = st.selectbox(
        "이미지 크기",
        [416, 512, 640],
        index=(
            [416, 512, 640].index(st.session_state.training_config.get("imgsz", 640))
            if st.session_state.training_config.get("imgsz") in [416, 512, 640]
            else 2
        ),
        help="학습에 사용할 이미지 크기(픽셀)를 설정합니다...",  # 기존 help 텍스트 유지
    )
with col4:
    optimizer = st.selectbox(
        "Optimizer",
        ["SGD", "Adam"],
        index=(
            ["SGD", "Adam"].index(
                st.session_state.training_config.get("optimizer", "Adam")
            )
            if st.session_state.training_config.get("optimizer") in ["SGD", "Adam"]
            else 1
        ),
        help="학습에 사용할 최적화 알고리즘을 선택합니다...",  # 기존 help 텍스트 유지
    )
with col5:
    learning_rate = st.number_input(
        "Learning Rate",
        value=float(st.session_state.training_config.get("learning_rate", 0.001)),
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

# 디바이스 선택 옵션 아래에 덮어쓰기 옵션 추가
overwrite = st.checkbox(
    "기존 결과 덮어쓰기",
    value=False,
    help="체크하면 동일한 프로젝트 이름의 기존 결과를 덮어씁니다.\n체크하지 않으면 증분된 이름으로 저장됩니다.",
)

# 프로젝트 이름 설정
project_name = st.text_input(
    "프로젝트 이름",
    value=st.session_state.training_config.get("project_name", "ocr_digit"),
    help="학습 결과가 저장될 프로젝트 폴더의 이름입니다...",  # 기존 help 텍스트 유지
)

# 학습 조건 미리보기 (학습 시작 버튼 위에 배치)
st.markdown("---")
st.subheader("📋 학습 실행 조건")
with st.expander("현재 설정된 학습 조건 확인", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"""
        **기본 설정**
        - 모델: {model_arch}
        - 프로젝트명: {project_name}
        - 학습 장치: {device.upper()}
        - 저장 경로: {paths_config.get('results_dir', 'runs/detect')}
        - 덮어쓰기: {'✅' if overwrite else '❌'}
        """
        )

    with col2:
        st.info(
            f"""
        **학습 파라미터**
        - Epochs: {epochs}회
        - Batch Size: {batch}
        - 이미지 크기: {imgsz}px
        - Optimizer: {optimizer}
        - Learning Rate: {learning_rate}
        """
        )

    # 데이터셋 정보 표시
    yaml_path = os.path.join("dataset", "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            st.info(
                f"""
            **데이터셋 정보**
            - 클래스 수: {data.get('nc', '정보없음')}개
            - 클래스명: {', '.join(data.get('names', ['정보없음']))}
            - 저장 경로: {os.path.abspath(RESULTS_DIR)}
            """
            )

    # YOLO 실행 명령어 미리보기 추가
    st.subheader("⌨️ YOLO 실행 명령어")
    data_yaml = os.path.abspath("dataset/data.yaml")
    yolo_cmd = (
        f"yolo task=detect mode=train model={model_arch} data={data_yaml} "
        f"epochs={epochs} batch={batch} imgsz={imgsz} lr0={learning_rate} "
        f"optimizer={optimizer.lower()} project={RESULTS_DIR} name={project_name} "
        f"device={device} exist_ok={str(overwrite).lower()} verbose=True"
    )
    st.code(yolo_cmd, language="bash")


# 학습 프로세스 확인 함수
def check_training_process():
    """현재 실행 중인 YOLO 학습 프로세스를 확인합니다."""
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if proc.info["cmdline"] and "yolo" in " ".join(proc.info["cmdline"]):
                return proc
    except:
        return None
    return None


# 학습 결과 분석 함수 추가
def show_training_results(project_name):
    """Display training results and all result images."""
    results_dir = os.path.join(RESULTS_DIR, project_name)
    results_path = os.path.join(results_dir, "results.csv")

    if not os.path.exists(results_dir):
        st.warning(f"⚠️ Cannot find results folder: {results_dir}")
        return

    # Results Plots
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)

        col1, col2 = st.columns(2)
        with col1:
            # Loss plot
            fig_loss = plt.figure(figsize=(10, 4))
            if "train/box_loss" in results_df.columns:
                plt.plot(
                    results_df.index, results_df["train/box_loss"], label="Box Loss"
                )
            if "train/cls_loss" in results_df.columns:
                plt.plot(
                    results_df.index, results_df["train/cls_loss"], label="Class Loss"
                )
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            st.pyplot(fig_loss)
            plt.close()

        with col2:
            # mAP plot
            if (
                "metrics/mAP50(B)" in results_df.columns
                or "val/mAP50" in results_df.columns
            ):
                fig_map = plt.figure(figsize=(10, 4))
                if "metrics/mAP50(B)" in results_df.columns:
                    plt.plot(
                        results_df.index, results_df["metrics/mAP50(B)"], label="mAP50"
                    )
                elif "val/mAP50" in results_df.columns:
                    plt.plot(results_df.index, results_df["val/mAP50"], label="mAP50")
                plt.title("Validation mAP")
                plt.xlabel("Epoch")
                plt.ylabel("mAP")
                plt.legend()
                st.pyplot(fig_map)
                plt.close()

    # Show all result images in the folder
    st.subheader("Result Images")
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(glob.glob(os.path.join(results_dir, ext)))

    if image_files:
        num_cols = 4
        for i in range(0, len(image_files), num_cols):
            cols = st.columns(num_cols)
            for j, col in enumerate(cols):
                if i + j < len(image_files):
                    img_path = image_files[i + j]
                    img = Image.open(img_path)
                    col.image(img, caption=os.path.basename(img_path))
    else:
        st.info("No result images found.")


# 페이지 로드 시 학습 상태 확인
if "training_pid" not in st.session_state:
    st.session_state.training_pid = None

# 학습 중인 프로세스 확인
training_process = check_training_process()
if training_process:
    st.warning(
        f"""⚠️ 이미 실행 중인 학습이 감지되었습니다!
    - PID: {training_process.pid}
    - 명령어: {' '.join(training_process.cmdline())}
    """
    )

    if st.button("❌ 기존 학습 중단"):
        try:
            training_process.kill()
            st.success("✅ 이전 학습이 중단되었습니다.")
        except Exception as e:
            st.error(f"학습 중단 실패: {str(e)}")

# 학습 시작 버튼
if st.button("🚀 학습 시작", use_container_width=True):
    try:
        # 이미 실행 중인 학습 확인
        if check_training_process():
            st.error(
                "❌ 이미 실행 중인 학습이 있습니다. 먼저 중단하고 다시 시도해주세요."
            )
            st.stop()

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
            f"device={device} exist_ok={str(overwrite).lower()} verbose=True"  # exist_ok 옵션 추가
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
                st.session_state.training_pid = None  # 학습 완료
                # 학습 결과 저장
                save_path = os.path.join(
                    RESULTS_DIR, project_name, "weights", "best.pt"
                )
                results_path = os.path.join(RESULTS_DIR, project_name, "results.csv")

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

                    # 학습 결과 분석 섹션
                    st.subheader("📊 학습 결과 분석")

                    # 결과 그래프 표시
                    if os.path.exists(results_path):
                        results_df = pd.read_csv(results_path)

                        # 손실 그래프
                        fig_loss = plt.figure(figsize=(10, 4))
                        plt.plot(
                            results_df.index, results_df["box_loss"], label="Box Loss"
                        )
                        plt.plot(
                            results_df.index, results_df["cls_loss"], label="Class Loss"
                        )
                        plt.title("학습 손실 그래프")
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.legend()
                        st.pyplot(fig_loss)
                        plt.close()

                        # mAP 그래프
                        if "metrics/mAP50(B)" in results_df.columns:
                            fig_map = plt.figure(figsize=(10, 4))
                            plt.plot(
                                results_df.index,
                                results_df["metrics/mAP50(B)"],
                                label="mAP50",
                            )
                            if "metrics/mAP50-95(B)" in results_df.columns:
                                plt.plot(
                                    results_df.index,
                                    results_df["metrics/mAP50-95(B)"],
                                    label="mAP50-95",
                                )
                            plt.title("검증 정확도 그래프")
                            plt.xlabel("Epoch")
                            plt.ylabel("mAP")
                            plt.legend()
                            st.pyplot(fig_map)
                            plt.close()

                    # 학습 결과 이미지 표시
                    results_img_path = os.path.join(RESULTS_DIR, project_name)
                    col1, col2 = st.columns(2)

                    # confusion matrix
                    confusion_matrix = os.path.join(
                        results_img_path, "confusion_matrix.png"
                    )
                    if os.path.exists(confusion_matrix):
                        with col1:
                            st.image(confusion_matrix, caption="Confusion Matrix")

                    # results
                    results_plot = os.path.join(results_img_path, "results.png")
                    if os.path.exists(results_plot):
                        with col2:
                            st.image(results_plot, caption="Results Plot")

                else:
                    st.warning("⚠️ 학습은 완료되었으나 모델 파일을 찾을 수 없습니다.")
            else:
                st.session_state.training_pid = None  # 학습 실패
                st.error("❌ 학습 중 오류가 발생했습니다.")
                if result.stderr:
                    st.code(result.stderr.decode(), language="bash")

    except Exception as e:
        st.session_state.training_pid = None  # 오류 발생
        st.error(f"❌ 오류 발생: {str(e)}")

# 메인 UI에 결과 분석 섹션 추가 (학습 시작 버튼 다음에 배치)
st.markdown("---")
st.subheader("🔍 이전 학습 결과 확인")

# 결과 폴더에서 프로젝트 목록 가져오기
available_projects = [
    d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))
]

if available_projects:
    selected_project = st.selectbox(
        "프로젝트 선택",
        available_projects,
        index=(
            available_projects.index(project_name)
            if project_name in available_projects
            else 0
        ),
    )
    show_training_results(selected_project)
else:
    st.info("💡 아직 학습된 결과가 없습니다.")
