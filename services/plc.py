# app/services/plc.py
import time
import streamlit as st


def send_to_plc(ip, port, register, value, max_retries=2):
    """
    감지된 값을 PLC로 전송하는 함수

    Args:
        ip (str): PLC IP 주소
        port (int): PLC 포트 번호
        register (int): PLC 레지스터 주소
        value (str): 전송할 값
        max_retries (int): 최대 재시도 횟수

    Returns:
        bool: 전송 성공 여부
    """
    try:
        # 실제 PLC 통신 구현 (여기서는 예시)
        # 실제로는 Modbus-TCP나 다른 프로토콜 라이브러리 사용

        # 임시 예시 (실제 구현 필요)
        for attempt in range(max_retries + 1):
            try:
                # PLC 통신 시뮬레이션 (실제 구현 필요)
                # from pymodbus.client.sync import ModbusTcpClient
                # client = ModbusTcpClient(ip, port=port)
                # client.write_register(register, int(value))
                # client.close()

                # 임시 성공 메시지
                st.success(
                    f"PLC 전송 성공: {value} -> {ip}:{port}, 레지스터 {register}"
                )
                return True

            except Exception as e:
                if attempt < max_retries:
                    st.warning(f"PLC 통신 실패, 재시도 {attempt+1}/{max_retries}...")
                    time.sleep(1)  # 재시도 전 대기
                else:
                    st.error(f"PLC 통신 최종 실패: {str(e)}")
                    return False

    except Exception as e:
        st.error(f"PLC 통신 오류: {str(e)}")
        return False
