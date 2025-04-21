import pandas as pd
import os
from datetime import datetime

LOG_PATH = "logs/detections.csv"
TRAIN_LOG_PATH = "logs/train_log.csv"
PLC_LOG_PATH = "logs/plc_sent.csv"


def log_detection(timestamp, digits):
    """감지된 숫자를 로그로 저장"""
    df = pd.DataFrame([{"timestamp": timestamp, "digits": ",".join(digits)}])
    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, mode="w", header=True, index=False)


def log_training(start_time, end_time, params, status, message=""):
    """모델 학습 기록 저장"""
    record = {
        "start_time": start_time,
        "end_time": end_time,
        "epochs": params.get("epochs"),
        "batch": params.get("batch"),
        "dataset": params.get("dataset", "uploaded"),
        "status": status,
        "message": message,
    }
    df = pd.DataFrame([record])
    if os.path.exists(TRAIN_LOG_PATH):
        df.to_csv(TRAIN_LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(TRAIN_LOG_PATH, mode="w", header=True, index=False)


def log_plc_transmission(timestamp, number, status):
    """PLC 전송 결과 기록 저장"""
    with open(PLC_LOG_PATH, "a") as f:
        f.write(f"{timestamp},{number},{status}\n")
