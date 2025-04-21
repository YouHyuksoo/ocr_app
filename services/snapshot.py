# app/services/snapshot.py
import os
import cv2
from datetime import datetime


def save_snapshot(frame, detections, label):
    """
    감지 결과가 표시된 스냅샷 이미지를 저장합니다.

    Args:
        frame: 원본 카메라 프레임
        detections: 감지된 객체 목록
        label: 이미지 파일명에 포함될 레이블(숫자)

    Returns:
        tuple: (저장된 파일 경로, 주석이 달린 프레임)
    """
    # 스냅샷 저장 디렉토리 생성
    snapshot_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "snapshots"
    )
    os.makedirs(snapshot_dir, exist_ok=True)

    # 파일명 생성 (현재 시간 + 레이블)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{label}.jpg"
    save_path = os.path.join(snapshot_dir, filename)

    # 감지 결과가 표시된 이미지 생성
    annotated_frame = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        confidence = d["confidence"]
        class_name = d["name"]

        # 박스 그리기
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 레이블 텍스트 표시
        label_text = f"{class_name} ({confidence:.2f})"
        cv2.putText(
            annotated_frame,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    # 이미지 파일로 저장
    cv2.imwrite(save_path, annotated_frame)

    # 파일 경로와 주석이 달린 프레임 반환
    return save_path, annotated_frame


def load_latest_snapshots(count=5):
    """
    최근 스냅샷 이미지를 불러옵니다.

    Args:
        count (int): 불러올 최근 스냅샷 수

    Returns:
        list: 최근 스냅샷 파일 경로 목록
    """
    snapshot_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "snapshots"
    )

    # 디렉토리가 없으면 빈 목록 반환
    if not os.path.exists(snapshot_dir):
        return []

    # 모든 jpg 파일 가져오기
    files = [
        os.path.join(snapshot_dir, f)
        for f in os.listdir(snapshot_dir)
        if f.endswith(".jpg")
    ]

    # 파일 수정 시간 기준으로 정렬
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # 최근 스냅샷 반환
    return files[:count]
