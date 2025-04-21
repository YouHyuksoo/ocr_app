import subprocess
import streamlit as st


def run_training(yolo_cmd):
    """
    YOLO 학습 명령어 실행 및 로그 출력
    """
    process = subprocess.Popen(
        yolo_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",  # 💡 이 줄 추가
    )
    log_output = []
    for line in process.stdout:
        log_output.append(line.strip())
        st.text(line.strip())
    process.wait()
    return subprocess.CompletedProcess(
        args=yolo_cmd, returncode=process.returncode, stdout="\n".join(log_output)
    )
