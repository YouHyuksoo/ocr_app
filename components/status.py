# app/components/status.py
import streamlit as st


class StatusBar:
    """
    앱의 현재 상태를 관리하고 표시하는 클래스
    """

    def __init__(self):
        """
        상태 바 초기화
        """
        self.container = st.empty()
        self.current_status = "초기화 중..."
        self._display()

    def update(self, status):
        """
        상태 메시지 업데이트

        Args:
            status (str): 표시할 상태 메시지
        """
        self.current_status = status
        self._display()

    def _display(self):
        """
        현재 상태를 화면에 표시
        """
        self.container.markdown(f"**상태:** {self.current_status}")

    def warning(self, message):
        """
        경고 메시지 표시

        Args:
            message (str): 경고 메시지
        """
        self.container.warning(message)
        self.current_status = message

    def error(self, message):
        """
        오류 메시지 표시

        Args:
            message (str): 오류 메시지
        """
        self.container.error(message)
        self.current_status = message

    def success(self, message):
        """
        성공 메시지 표시

        Args:
            message (str): 성공 메시지
        """
        self.container.success(message)
        self.current_status = message
