import cv2
import torch
import streamlit as st
from ultralytics import YOLO
import os


def setup_detector(model_path):
    if not os.path.exists(model_path):
        st.error(f"❌ 모델 파일 없음: {model_path}")
        st.stop()

    try:
        model = YOLO(model_path, task="detect")
        model.overrides["conf"] = 0.25
        model.overrides["iou"] = 0.45
        model.overrides["agnostic_nms"] = False
        model.overrides["max_det"] = 10
        return model
    except Exception as e:
        st.error(f"❌ 모델 로드 중 오류 발생: {e}")
        st.stop()


def process_detections(
    model, frame, confidence_threshold, is_digit_mode, cfg, status_bar
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

        if is_digit_mode and detected_digits:
            combined = "".join(sorted(detected_digits))

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
