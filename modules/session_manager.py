from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import streamlit as st
import uuid


class SessionManager:
    """
    Manages user sessions and chat history for Streamlit applications.
    """
    
    @staticmethod
    def initialize_session() -> None:
        """
        Initialize session state variables if they don't exist.
        """
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'ocr_results' not in st.session_state:
            st.session_state.ocr_results = None
        
        if 'rag_initialized' not in st.session_state:
            st.session_state.rag_initialized = False
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = {}
        
        if 'user_settings' not in st.session_state:
            st.session_state.user_settings = {
                'theme': 'light',
                'temperature': 0.2,
                'max_tokens': 2048
            }
    
    @staticmethod
    def add_message(role: str, content: str) -> None:
        """
        Add a message to the chat history.
        
        Args:
            role: The role of the message sender ('user' or 'assistant')
            content: The content of the message
        """
        st.session_state.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "message_id": str(uuid.uuid4())
        })
    
    @staticmethod
    def get_chat_history() -> List[Dict[str, Any]]:
        """
        Get the chat history.
        
        Returns:
            The chat history as a list of message dictionaries
        """
        return st.session_state.chat_history
    
    @staticmethod
    def get_chat_history_for_rag() -> List[Tuple[str, str]]:
        """
        Get the chat history in a format suitable for RAG.
        
        Returns:
            The chat history as a list of tuples (human_message, ai_message)
        """
        history = []
        messages = st.session_state.chat_history
        
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                    history.append((messages[i]["content"], messages[i+1]["content"]))
        
        return history
    
    @staticmethod
    def clear_chat_history() -> None:
        """
        Clear the chat history.
        """
        st.session_state.chat_history = []
    
    @staticmethod
    def store_ocr_results(results: str) -> None:
        """
        Store OCR results in the session.
        
        Args:
            results: The OCR results
        """
        st.session_state.ocr_results = results
    
    @staticmethod
    def get_ocr_results() -> Optional[str]:
        """
        Get the OCR results from the session.
        
        Returns:
            The OCR results or None if not available
        """
        return st.session_state.ocr_results
    
    @staticmethod
    def set_rag_initialized(value: bool = True) -> None:
        """
        Set the RAG initialization status.
        
        Args:
            value: The initialization status
        """
        st.session_state.rag_initialized = value
    
    @staticmethod
    def is_rag_initialized() -> bool:
        """
        Check if RAG is initialized.
        
        Returns:
            True if RAG is initialized, False otherwise
        """
        return st.session_state.rag_initialized
    
    @staticmethod
    def store_uploaded_file(file_id: str, file_info: Dict[str, Any]) -> None:
        """
        Store information about an uploaded file.
        
        Args:
            file_id: The ID of the file
            file_info: Information about the file
        """
        st.session_state.uploaded_files[file_id] = file_info
    
    @staticmethod
    def get_uploaded_files() -> Dict[str, Dict[str, Any]]:
        """
        Get information about all uploaded files.
        
        Returns:
            Information about all uploaded files
        """
        return st.session_state.uploaded_files
    
    @staticmethod
    def update_user_setting(setting_name: str, value: Any) -> None:
        """
        Update a user setting.
        
        Args:
            setting_name: The name of the setting to update
            value: The new value for the setting
        """
        if 'user_settings' in st.session_state:
            st.session_state.user_settings[setting_name] = value
    
    @staticmethod
    def get_user_setting(setting_name: str, default: Any = None) -> Any:
        """
        Get a user setting.
        
        Args:
            setting_name: The name of the setting to get
            default: The default value to return if the setting doesn't exist
            
        Returns:
            The value of the setting or the default value
        """
        if 'user_settings' in st.session_state and setting_name in st.session_state.user_settings:
            return st.session_state.user_settings[setting_name]
        return default