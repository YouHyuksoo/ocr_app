import cv2
import torch
import streamlit as st
from ultralytics import YOLO
import os


def setup_detector(model_path, detection_settings=None):
    """
    YOLO 모델을 설정값과 함께 초기화합니다.

    Args:
        model_path (str): YOLO 모델 파일 경로
        detection_settings (dict): 모델 설정값 딕셔너리
            - conf: 신뢰도 임계값 (기본값: 0.25)
            - iou: IoU 임계값 (기본값: 0.45)
            - agnostic_nms: 클래스 독립적 NMS 여부 (기본값: False)
            - max_det: 최대 감지 개수 (기본값: 10)
    """
    if not os.path.exists(model_path):
        st.error(f"❌ 모델 파일 없음: {model_path}")
        st.stop()

    try:
        # 기본 설정값 정의
        settings = {"conf": 0.25, "iou": 0.45, "agnostic_nms": False, "max_det": 10}

        # 전달받은 설정값으로 업데이트
        if detection_settings:
            settings.update(
                {
                    k: v
                    for k, v in detection_settings.items()
                    if k in ["conf", "iou", "agnostic_nms", "max_det"]
                }
            )

        # 모델 초기화
        model = YOLO(model_path, task="detect")

        # 설정값 적용
        model.overrides["conf"] = settings["conf"]
        model.overrides["iou"] = settings["iou"]
        model.overrides["agnostic_nms"] = settings["agnostic_nms"]
        model.overrides["max_det"] = settings["max_det"]

        return model

    except Exception as e:
        st.error(f"❌ 모델 로드 중 오류 발생: {e}")
        st.stop()


def sort_digits_by_direction(detections, direction):
    """진입 방향에 따라 감지된 숫자들을 정렬"""
    if not detections:
        return []

    # 각 감지에 대해 중심 좌표 계산
    digits_with_centers = []
    for det in detections:
        x1, y1, x2, y2 = det.boxes.xyxy[0][:4].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        class_id = int(det.boxes.cls[0].item())
        digits_with_centers.append((class_id, center_x, center_y))

    # 방향에 따른 정렬
    if direction == "left_to_right":
        sorted_digits = sorted(
            digits_with_centers, key=lambda x: x[1]
        )  # x좌표 오름차순
    elif direction == "right_to_left":
        sorted_digits = sorted(
            digits_with_centers, key=lambda x: x[1], reverse=True
        )  # x좌표 내림차순
    elif direction == "top_to_bottom":
        sorted_digits = sorted(
            digits_with_centers, key=lambda x: x[2]
        )  # y좌표 오름차순
    else:  # bottom_to_top
        sorted_digits = sorted(
            digits_with_centers, key=lambda x: x[2], reverse=True
        )  # y좌표 내림차순

    return [str(d[0]) for d in sorted_digits]


def process_detections(
    model,
    frame,
    confidence_threshold,
    is_digit_mode,
    detection_settings,  # 설정을 딕셔너리로 받도록 수정
    status_bar,
):
    try:
        status_bar.update("모델로 감지 수행 중...")
        results = model(frame)
        status_bar.update("감지 결과 처리 중...")

        detections = []
        detected_digits = []
        combined = None

        annotated_display = frame.copy()

        for box in results[0].boxes:
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            name = model.names[class_id]

            if confidence >= confidence_threshold:
                bbox = box.xyxy.tolist()[0]
                detections.append(
                    {"confidence": confidence, "name": name, "bbox": bbox}
                )

                if is_digit_mode and name.isdigit():
                    detected_digits.append(name)

                # 바운딩 박스 및 라벨 시각화
                x1, y1, x2, y2 = map(int, bbox)
                label = f"{name} ({confidence:.2f})"
                cv2.rectangle(annotated_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_display,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        if is_digit_mode and results:
            # 전달받은 설정 사용
            entry_direction = detection_settings.get("entry_direction", "left_to_right")
            expected_digits = detection_settings.get("digit_count", 3)

            # 감지된 숫자들을 방향에 따라 정렬
            detected_digits = sort_digits_by_direction(results[0], entry_direction)

            # 자릿수 체크
            if len(detected_digits) != expected_digits:
                status_bar.update(
                    f"⚠️ 예상 자릿수({expected_digits})와 다름: {len(detected_digits)}자리 감지됨"
                )
            else:
                status_bar.update("✅ 정상 감지됨")

            combined = "".join(detected_digits)

        status_bar.update(f"감지 완료 - 총 {len(detections)}개 객체 감지됨")

        print(f"[디버깅] YOLO 감지 객체 수: {len(results[0].boxes)}")
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            name = model.names[class_id]
            conf = box.conf.item()
            print(f" - 클래스: {name}, 신뢰도: {conf}")

        return detections, detected_digits, combined, annotated_display

    except Exception as e:
        status_bar.update(f"감지 중 오류 발생: {e}")
        raise
