import os
import tempfile
import re
from typing import Optional, Union, BinaryIO
from pathlib import Path

from mistralai import Mistral
import logging

logger = logging.getLogger(__name__)

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
        
        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-ocr-latest"
    
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
            # Upload the file to Mistral
            with open(temp_file_path, "rb") as file:
                uploaded_file = self.client.files.upload(
                    file={
                        "file_name": "uploaded_image.png",
                        "content": file,
                    },
                    purpose="ocr"
                )
            
            # Get signed URL for the uploaded file
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
            
            # Process the document with Mistral OCR
            ocr_response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                }
            )
            
            # Extract text from the document
            # The OCR response object might have different structure depending on the API version
            # Log detailed information about the OCR response for debugging
            logger.debug(f"OCR Response type: {type(ocr_response)}")
            logger.debug(f"OCR Response attributes: {dir(ocr_response)}")
            
            # Check for mistralai.models.ocrresponse.OCRResponse type
            if hasattr(ocr_response, 'pages') and isinstance(ocr_response.pages, list):
                # Extract text from all pages and join them
                all_text = []
                for page in ocr_response.pages:
                    # First check for markdown attribute (newer API version)
                    if hasattr(page, 'markdown'):
                        all_text.append(page.markdown)
                    # Fallback to text attribute for compatibility
                    elif hasattr(page, 'text'):
                        all_text.append(page.text)
                
                if all_text:
                    return "\n\n".join(all_text)
            
            # Try other possible response structures
            if hasattr(ocr_response, 'text'):
                return ocr_response.text
            elif hasattr(ocr_response, 'document') and hasattr(ocr_response.document, 'text'):
                return ocr_response.document.text
            elif hasattr(ocr_response, 'content'):
                return ocr_response.content
            else:
                # Fallback: try to access response as dictionary
                try:
                    if isinstance(ocr_response, dict):
                        if 'text' in ocr_response:
                            return ocr_response['text']
                        elif 'document' in ocr_response and 'text' in ocr_response['document']:
                            return ocr_response['document']['text']
                        elif 'content' in ocr_response:
                            return ocr_response['content']
                        elif 'pages' in ocr_response and isinstance(ocr_response['pages'], list):
                            all_text = []
                            for page in ocr_response['pages']:
                                if 'markdown' in page:
                                    all_text.append(page['markdown'])
                                elif 'text' in page:
                                    all_text.append(page['text'])
                            if all_text:
                                return "\n\n".join(all_text)
                except Exception as e:
                    logger.debug(f"Error trying to access response as dictionary: {str(e)}")
                
                # If we get here, we need to inspect the response structure
                logger.info(f"OCR Response structure: {type(ocr_response)}, attrs: {dir(ocr_response)}")
                raise ValueError(f"Could not extract text from OCR response. Response type: {type(ocr_response)}")
        except Exception as e:
            logger.error(f"Error processing image with OCR: {str(e)}")
            raise
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
            # Upload the file to Mistral
            with open(temp_file_path, "rb") as file:
                uploaded_file = self.client.files.upload(
                    file={
                        "file_name": "uploaded_file.pdf",
                        "content": file,
                    },
                    purpose="ocr"
                )
            
            # Get signed URL for the uploaded file
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
            
            # Process the document with Mistral OCR
            ocr_response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                }
            )
            
            # Extract text from the document
            # The OCR response object might have different structure depending on the API version
            # Log detailed information about the OCR response for debugging
            logger.debug(f"OCR Response type: {type(ocr_response)}")
            logger.debug(f"OCR Response attributes: {dir(ocr_response)}")
            
            # Check for mistralai.models.ocrresponse.OCRResponse type
            if hasattr(ocr_response, 'pages') and isinstance(ocr_response.pages, list):
                # Extract text from all pages and join them
                all_text = []
                for page in ocr_response.pages:
                    # First check for markdown attribute (newer API version)
                    if hasattr(page, 'markdown'):
                        all_text.append(page.markdown)
                    # Fallback to text attribute for compatibility
                    elif hasattr(page, 'text'):
                        all_text.append(page.text)
                
                if all_text:
                    return "\n\n".join(all_text)
            
            # Try other possible response structures
            if hasattr(ocr_response, 'text'):
                return ocr_response.text
            elif hasattr(ocr_response, 'document') and hasattr(ocr_response.document, 'text'):
                return ocr_response.document.text
            elif hasattr(ocr_response, 'content'):
                return ocr_response.content
            else:
                # Fallback: try to access response as dictionary
                try:
                    if isinstance(ocr_response, dict):
                        if 'text' in ocr_response:
                            return ocr_response['text']
                        elif 'document' in ocr_response and 'text' in ocr_response['document']:
                            return ocr_response['document']['text']
                        elif 'content' in ocr_response:
                            return ocr_response['content']
                        elif 'pages' in ocr_response and isinstance(ocr_response['pages'], list):
                            all_text = []
                            for page in ocr_response['pages']:
                                if 'markdown' in page:
                                    all_text.append(page['markdown'])
                                elif 'text' in page:
                                    all_text.append(page['text'])
                            if all_text:
                                return "\n\n".join(all_text)
                except Exception as e:
                    logger.debug(f"Error trying to access response as dictionary: {str(e)}")
                
                # If we get here, we need to inspect the response structure
                logger.info(f"OCR Response structure: {type(ocr_response)}, attrs: {dir(ocr_response)}")
                raise ValueError(f"Could not extract text from OCR response. Response type: {type(ocr_response)}")
        except Exception as e:
            logger.error(f"Error processing PDF with OCR: {str(e)}")
            raise
        finally:
            # Clean up the temporary file
            Path(temp_file_path).unlink(missing_ok=True)
    
    def extract_text(self, file: BinaryIO, file_type: str, strip_markdown: bool = False) -> str:
        """
        Extract text from a file using the appropriate method based on file type.
        
        Args:
            file: The uploaded file object
            file_type: The type of the file ('pdf', 'image', etc.)
            strip_markdown: If True, attempt to strip Markdown formatting from the result
            
        Returns:
            Extracted text from the file
            
        Raises:
            ValueError: If the file type is not supported
        """
        if file_type.lower() == 'pdf':
            text = self.process_pdf(file)
        elif file_type.lower() in ['image', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
            text = self.process_image(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        if strip_markdown and text:
            # Simple Markdown removal - this can be enhanced if needed
            # Remove headers
            text = re.sub(r'#{1,6}\s+', '', text)
            # Remove bold/italic
            text = re.sub(r'\*\*|\*|__|\|', '', text)
            # Remove links but keep the text
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
            # Remove code blocks but keep content
            text = re.sub(r'```[a-z]*\n|```', '', text)
            # Remove single line code
            text = re.sub(r'`([^`]+)`', r'\1', text)
            # Remove bullet points
            text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
            # Remove numbered lists
            text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
            
        return text
    
    def extract_text_from_ocr_response(self, ocr_response) -> str:
        """
        Extract text from the OCR response object.
        
        Args:
            ocr_response: The OCR response object from Mistral AI
            
        Returns:
            Extracted text from the response
            
        Raises:
            ValueError: If text cannot be extracted from the response
        """
        # Log detailed information about the OCR response for debugging
        logger.debug(f"OCR Response type: {type(ocr_response)}")
        logger.debug(f"OCR Response attributes: {dir(ocr_response)}")
        
        # Check for mistralai.models.ocrresponse.OCRResponse type
        if hasattr(ocr_response, 'pages') and isinstance(ocr_response.pages, list):
            # Extract text from all pages and join them
            all_text = []
            for page in ocr_response.pages:
                # First check for markdown attribute (newer API version)
                if hasattr(page, 'markdown'):
                    all_text.append(page.markdown)
                # Fallback to text attribute for compatibility
                elif hasattr(page, 'text'):
                    all_text.append(page.text)
            
            if all_text:
                return "\n\n".join(all_text)
        
        # Try other possible response structures
        if hasattr(ocr_response, 'text'):
            return ocr_response.text
        elif hasattr(ocr_response, 'document') and hasattr(ocr_response.document, 'text'):
            return ocr_response.document.text
        elif hasattr(ocr_response, 'content'):
            return ocr_response.content
        else:
            # Fallback: try to access response as dictionary
            try:
                if isinstance(ocr_response, dict):
                    if 'text' in ocr_response:
                        return ocr_response['text']
                    elif 'document' in ocr_response and 'text' in ocr_response['document']:
                        return ocr_response['document']['text']
                    elif 'content' in ocr_response:
                        return ocr_response['content']
                    elif 'pages' in ocr_response and isinstance(ocr_response['pages'], list):
                        all_text = []
                        for page in ocr_response['pages']:
                            if 'markdown' in page:
                                all_text.append(page['markdown'])
                            elif 'text' in page:
                                all_text.append(page['text'])
                        if all_text:
                            return "\n\n".join(all_text)
            except Exception as e:
                logger.debug(f"Error trying to access response as dictionary: {str(e)}")
            
            # If we get here, we need to inspect the response structure
            logger.info(f"OCR Response structure: {type(ocr_response)}, attrs: {dir(ocr_response)}")
            if hasattr(ocr_response, 'pages'):
                logger.info(f"Pages attribute exists with {len(ocr_response.pages)} pages")
                if len(ocr_response.pages) > 0:
                    logger.info(f"First page attributes: {dir(ocr_response.pages[0])}")
            raise ValueError(f"Could not extract text from OCR response. Response type: {type(ocr_response)}")