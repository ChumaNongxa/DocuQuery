# OCR Processor

A modern Python application for extracting text from images and PDF documents using Mistral AI's OCR capabilities and enabling document Q&A with Google's Gemini models.

![UI Screenshot](Other/UI%20Screenshot.png)

## Overview

OCR Processor is a powerful document processing application that combines OCR (Optical Character Recognition) with RAG (Retrieval-Augmented Generation) to enable users to extract text from documents and have interactive conversations about their content.

## Project Structure

```
ocr-processor/
├── app.py                 # Main Streamlit application
├── modules/               # Package containing all modules
│   ├── __init__.py        # Makes modules a proper package
│   ├── ocr_processor.py   # OCR processing functionality
│   ├── rag_processor.py   # RAG processing functionality
│   └── session_manager.py # Session management for Streamlit
├── .env.example           # Example environment variables
├── README.md              # This documentation
└── requirements.txt       # Project dependencies
```

## Features

- Extract text from image files (PNG, JPEG, etc.) using Mistral AI's OCR
- Extract text from PDF documents using Mistral AI's OCR
- Extract text from Word documents (DOCX)
- RAG (Retrieval-Augmented Generation) for document Q&A using Google's Gemini models
- Modern Streamlit web interface with chat functionality
- Session management for persistent chat history
- Type hints and modern Python 3.11+ features
- Environment variable configuration with dotenv
- Comprehensive error handling and logging

## Requirements

- Python 3.11 or higher
- Mistral AI API key
- Google API key (for Gemini models)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ocr-processor.git
   cd ocr-processor
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   cp .env.example .env
   ```
   Then edit the .env file with your API keys.

## Usage

### Running the Streamlit App

1. Make sure your virtual environment is activated and your API keys are set in the .env file.

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. The app will be available at http://localhost:8501 by default.

### Using the OCR Processor in Your Code

```python
from modules.ocr_processor import OCRProcessor

# Initialize with API key from environment variable
processor = OCRProcessor()

# Or provide API key directly
# processor = OCRProcessor(api_key="your-api-key-here")

# Process an image file
with open("document.png", "rb") as image_file:
    text = processor.extract_text(image_file, file_type="image")
    print(text)

# Process a PDF file
with open("document.pdf", "rb") as pdf_file:
    text = processor.extract_text(pdf_file, file_type="pdf")
    print(text)
```

### Using the RAG Processor in Your Code

```python
from modules.rag_processor import RAGProcessor

# Initialize with API key from environment variable
processor = RAGProcessor()

# Process a document
with open("document.pdf", "rb") as pdf_file:
    processor.process_document(file=pdf_file, file_type="pdf")

# Or process text directly
processor.process_document(text="Your document text here")

# Query the document
response = processor.query("What is this document about?")
print(response["answer"])
```

## API Reference

### OCRProcessor

The OCRProcessor class provides methods for extracting text from images and PDFs using Mistral AI's OCR capabilities.

### RAGProcessor

The RAGProcessor class provides methods for processing documents and answering questions about them using Google's Gemini models.

### SessionManager

The SessionManager class provides methods for managing user sessions and chat history in Streamlit applications.

## Development

### Adding New Features

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Implement your changes
4. Add tests if applicable
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## License

[MIT License](LICENSE)

## Acknowledgements

- [Mistral AI](https://mistral.ai/) for their OCR capabilities
- [Google Gemini](https://ai.google.dev/) for their language models
- [Streamlit](https://streamlit.io/) for the web interface
- [LangChain](https://www.langchain.com/) for the RAG implementation