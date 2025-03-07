import os
import tempfile
from typing import Optional, Union, BinaryIO
from pathlib import Path

from mistralai.client import MistralClient
from mistralai.models.documents import DocumentProcessingRequest
from PIL import Image
import PyPDF2
import io


class OCRProcessor:
    """
    A processor for extracting text from images and PDFs using Mistral AI's OCR capabilities.
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the OCR processor with Mistral AI.
        
        Args:
            api_key: Mistral AI API key. If None, it will try to get from environment variable.
            
        Raises:
            ValueError: If no API key is provided or found in environment variables.
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set it as an environment variable or pass it directly.")
        
        self.client = MistralClient(api_key=self.api_key)
    
    def process_image(self, image_file: BinaryIO) -> str:
        """
        Process an image file using Mistral OCR.
        
        Args:
            image_file: The uploaded image file object
            
        Returns:
            Extracted text from the image
            
        Raises:
            Exception: If there's an error processing the image
        """
        # Create a temporary file to store the image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file_path = temp_file.name
            # Save the uploaded file to the temporary file
            image_file.seek(0)
            temp_file.write(image_file.read())
        
        try:
            # Process the document with Mistral OCR
            request = DocumentProcessingRequest(file_path=temp_file_path)
            document = self.client.process_document(request)
            
            # Extract text from the document
            extracted_text = document.text
            return extracted_text
        finally:
            # Clean up the temporary file
            Path(temp_file_path).unlink(missing_ok=True)
    
    def process_pdf(self, pdf_file: BinaryIO) -> str:
        """
        Process a PDF file using Mistral OCR.
        
        Args:
            pdf_file: The uploaded PDF file object
            
        Returns:
            Extracted text from the PDF
            
        Raises:
            Exception: If there's an error processing the PDF
        """
        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file_path = temp_file.name
            # Save the uploaded file to the temporary file
            pdf_file.seek(0)
            temp_file.write(pdf_file.read())
        
        try:
            # Process the document with Mistral OCR
            request = DocumentProcessingRequest(file_path=temp_file_path)
            document = self.client.process_document(request)
            
            # Extract text from the document
            extracted_text = document.text
            return extracted_text
        finally:
            # Clean up the temporary file
            Path(temp_file_path).unlink(missing_ok=True)
    
    def extract_text(self, file: BinaryIO, file_type: str) -> str:
        """
        Extract text from a file based on its type.
        
        Args:
            file: The uploaded file object
            file_type: The type of the file ('image' or 'pdf')
            
        Returns:
            Extracted text from the file
            
        Raises:
            ValueError: If the file type is not supported
            Exception: If there's an error processing the file
        """
        match file_type:
            case "image":
                return self.process_image(file)
            case "pdf":
                return self.process_pdf(file)
            case _:
                raise ValueError(f"Unsupported file type for OCR: {file_type}")