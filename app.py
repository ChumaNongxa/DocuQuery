# Import Libraries
import os
import re
import logging
import streamlit as st
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Import modules
from modules.ocr_processor import OCRProcessor
from modules.rag_processor import RAGProcessor
from modules.session_manager import SessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_page_config() -> None:
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Document Processing App",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "# Document Processing App\nExtract text from documents and chat with them using AI."
        },
    )


def setup_sidebar() -> Optional[tuple]:
    """
    Set up the sidebar for file upload and processing.

    Returns:
        A tuple of (file, file_type) if a file is uploaded and processed, None otherwise.
    """
    result = None
    
    with st.sidebar:
        # Document Upload Section
        st.header("📁 Upload Document")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "png", "jpg", "jpeg", "docx"],
            help="Upload a PDF, image, or Word document to extract text and chat with it.",
        )

        if uploaded_file is not None:
            file_type = None
            if uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg")):
                file_type = "image"
                st.image(
                    uploaded_file,
                    caption=f"Uploaded: {uploaded_file.name}",
                    use_container_width=True,
                )
            elif uploaded_file.name.lower().endswith(".pdf"):
                file_type = "pdf"
                st.info(f"Uploaded PDF: {uploaded_file.name}")
            elif uploaded_file.name.lower().endswith(".docx"):
                file_type = "docx"
                st.info(f"Uploaded Word document: {uploaded_file.name}")

            # Process button
            if st.button("🔍 Process Document", use_container_width=True):
                result = (uploaded_file, file_type)
        
        # Settings section - Always visible
        st.divider()
        st.header("⚙️ Settings")
        
        # Layout toggle
        previous_layout = SessionManager.get_user_setting("layout", "side-by-side")
        layout_option = st.radio(
            "Layout",
            options=["Side-by-side", "Stacked"],
            index=0 if previous_layout == "side-by-side" else 1,
            help="Choose how to display the extracted text and chat interface."
        )
        new_layout = "side-by-side" if layout_option == "Side-by-side" else "stacked"
        
        # Update layout setting and rerun if changed
        if previous_layout != new_layout:
            SessionManager.update_user_setting("layout", new_layout)
            st.rerun()

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=SessionManager.get_user_setting("temperature", 0.2),
            step=0.1,
            help="Higher values make the output more random, lower values make it more deterministic.",
        )
        SessionManager.update_user_setting("temperature", temperature)

        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=256,
            max_value=4096,
            value=SessionManager.get_user_setting("max_tokens", 2048),
            step=256,
            help="Maximum number of tokens to generate in the response.",
        )
        SessionManager.update_user_setting("max_tokens", max_tokens)

        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            SessionManager.clear_chat_history()
            st.success("Chat history cleared!")
            st.rerun()

    return result


def process_document(file, file_type) -> bool:
    """
    Process a document with OCR and RAG.

    Args:
        file: The uploaded file
        file_type: The type of the file

    Returns:
        True if processing was successful, False otherwise
    """
    try:
        with st.status("Processing document...", expanded=True) as status:
            # Process with OCR if it's an image or PDF
            status.update(label="Extracting text with OCR...")
            if file_type in ["image", "pdf"]:
                ocr_processor = OCRProcessor()
                extracted_text = ocr_processor.extract_text(file, file_type)
                SessionManager.store_ocr_results(extracted_text)
                status.update(label="OCR processing complete!")
            elif file_type == "docx":
                # For Word documents, use the RAG processor's extraction method
                rag_processor = RAGProcessor()
                extracted_text = rag_processor.extract_text_from_docx(file)
                SessionManager.store_ocr_results(extracted_text)
                status.update(label="Text extraction complete!")

            # Initialize RAG with the extracted text
            if SessionManager.get_ocr_results():
                status.update(label="Initializing RAG system...")
                rag_processor = RAGProcessor()
                rag_processor.process_document(text=SessionManager.get_ocr_results())
                SessionManager.set_rag_initialized(True)

                # Store the RAG processor in session state for later use
                st.session_state.rag_processor = rag_processor
                status.update(label="Document processing complete!", state="complete")
                return True
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        st.error(f"Error processing document: {str(e)}")

    return False


def display_chat_interface() -> None:
    """Display the chat interface."""
    # Get the current layout setting
    layout = SessionManager.get_user_setting("layout", "side-by-side")
    
    if layout == "side-by-side":
        # Side-by-side layout (original)
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.header("📝 Extracted Text")
            if SessionManager.get_ocr_results():
                st.text_area(
                    "OCR Results",
                    SessionManager.get_ocr_results(),
                    height=500,
                    disabled=True,
                )
            else:
                st.info("Upload and process a document to see extracted text.")
        
        with col2:
            st.header("💬 Chat with Document")
            
            # Create a container for chat messages
            chat_container = st.container()
            
            # Create a container for the input box that will always be at the bottom
            input_container = st.container()
            
            # Handle input in the bottom container
            with input_container:
                user_input = st.chat_input("Ask a question about the document...")
            
            # Display chat history in the chat container
            with chat_container:
                for message in SessionManager.get_chat_history():
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                
                # Process new input if provided
                if user_input:
                    # Add user message to chat history
                    SessionManager.add_message("user", user_input)
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.write(user_input)
                    
                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        if SessionManager.is_rag_initialized() and hasattr(
                            st.session_state, "rag_processor"
                        ):
                            with st.spinner("Thinking..."):
                                try:
                                    # Get response from RAG
                                    chat_history = SessionManager.get_chat_history_for_rag()
                                    response = st.session_state.rag_processor.query(
                                        user_input, chat_history
                                    )
                                    
                                    # Display response
                                    st.write(response["answer"])
                                    
                                    # Add assistant message to chat history
                                    SessionManager.add_message("assistant", response["answer"])
                                    
                                    # Display source documents in an expander
                                    if response["source_documents"]:
                                        with st.expander("Source Documents"):
                                            for i, doc in enumerate(
                                                response["source_documents"]
                                            ):
                                                st.markdown(f"**Source {i+1}:**")
                                                st.text(doc.page_content)
                                                st.divider()
                                except Exception as e:
                                    logger.error(
                                        f"Error generating response: {str(e)}", exc_info=True
                                    )
                                    error_message = f"Error generating response: {str(e)}"
                                    st.error(error_message)
                                    SessionManager.add_message("assistant", error_message)
                        else:
                            message = "Please upload and process a document first."
                            st.warning(message)
                            SessionManager.add_message("assistant", message)
    else:
        # Stacked layout
        # First section: Extracted Text
        st.header("📝 Extracted Text")
        if SessionManager.get_ocr_results():
            st.text_area(
                "OCR Results",
                SessionManager.get_ocr_results(),
                height=300,  # Reduced height for stacked layout
                disabled=True,
            )
        else:
            st.info("Upload and process a document to see extracted text.")
        
        # Add a divider between sections
        st.divider()
        
        # Second section: Chat with Document
        st.header("💬 Chat with Document")
        
        # Create a container for chat messages
        chat_container = st.container()
        
        # Create a container for the input box that will always be at the bottom
        input_container = st.container()
        
        # Handle input in the bottom container
        with input_container:
            user_input = st.chat_input("Ask a question about the document...")
        
        # Display chat history in the chat container
        with chat_container:
            for message in SessionManager.get_chat_history():
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Process new input if provided
            if user_input:
                # Add user message to chat history
                SessionManager.add_message("user", user_input)
                
                # Display user message
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    if SessionManager.is_rag_initialized() and hasattr(
                        st.session_state, "rag_processor"
                    ):
                        with st.spinner("Thinking..."):
                            try:
                                # Get response from RAG
                                chat_history = SessionManager.get_chat_history_for_rag()
                                response = st.session_state.rag_processor.query(
                                    user_input, chat_history
                                )
                                
                                # Display response
                                st.write(response["answer"])
                                
                                # Add assistant message to chat history
                                SessionManager.add_message("assistant", response["answer"])
                                
                                # Display source documents in an expander
                                if response["source_documents"]:
                                    with st.expander("Source Documents"):
                                        for i, doc in enumerate(
                                            response["source_documents"]
                                        ):
                                            st.markdown(f"**Source {i+1}:**")
                                            st.text(doc.page_content)
                                            st.divider()
                            except Exception as e:
                                logger.error(
                                    f"Error generating response: {str(e)}", exc_info=True
                                )
                                error_message = f"Error generating response: {str(e)}"
                                st.error(error_message)
                                SessionManager.add_message("assistant", error_message)
                    else:
                        message = "Please upload and process a document first."
                        st.warning(message)
                        SessionManager.add_message("assistant", message)


def main() -> None:
    """Main application function for the OCR and RAG system."""
    # Set up page configuration
    setup_page_config()

    # Initialize session
    SessionManager.initialize_session()
    
    # Ensure static/images directory exists
    os.makedirs(os.path.join("static", "images"), exist_ok=True)

    # Display title and description
    st.title("📄 Document Processing App")
    st.markdown(
        """
        Upload a document, extract text with OCR, and chat with it using AI.
        
        **Supported file types:**
        - Images (PNG, JPG, JPEG)
        - PDF documents
        - Word documents (DOCX)
        """
    )

    # Set up sidebar and get uploaded file
    result = setup_sidebar()

    # Process document if a file was uploaded
    if result:
        file, file_type = result
        success = process_document(file, file_type)
        if success:
            st.success(f"Successfully processed {file.name}!")

    # Display chat interface
    display_chat_interface()


if __name__ == "__main__":
    main()
