#!/usr/bin/env python3
"""
document_processor.py - Document processing with docling

This module provides functionality to process documents (URLs and PDFs) with docling
and convert them to markdown format.
"""

import os
import logging
from io import BytesIO
from typing import Dict, Optional, Union
from pathlib import Path

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TesseractOcrOptions,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("document_processor.log")
    ]
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing with docling"""
    
    def __init__(self):
        """Initialize the document processor"""
        # Configure accelerator options
        self.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.CPU
        )
        
        # Configure pipeline options
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.accelerator_options = self.accelerator_options
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        
        # Create document converter
        self.converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=self.pipeline_options,
            ),
            InputFormat.IMAGE: TesseractOcrOptions(),
            InputFormat.DOCX: None,
            InputFormat.HTML: None,
        })
        
        logger.info("Document processor initialized")
    
    def process_url(self, url: str) -> str:
        """
        Process a URL with docling and convert to markdown
        
        Args:
            url: URL to process
            
        Returns:
            Markdown text
        """
        logger.info(f"Processing URL: {url}")
        
        try:
            # Convert URL to markdown using docling
            result = self.converter.convert(url)
            
            # Export to markdown
            markdown_text = result.document.export_to_markdown()
            
            logger.info(f"Successfully processed URL: {url}")
            return markdown_text
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            raise Exception(f"Error processing URL: {str(e)}")
    
    def process_pdf(self, pdf_path: str) -> str:
        """
        Process a PDF file with docling and convert to markdown
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Markdown text
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Convert PDF to markdown using docling
            result = self.converter.convert(pdf_path)
            
            # Export to markdown
            markdown_text = result.document.export_to_markdown()
            
            logger.info(f"Successfully processed PDF: {pdf_path}")
            return markdown_text
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def process_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """
        Process PDF bytes with docling and convert to markdown
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Markdown text
        """
        logger.info("Processing PDF bytes")
        
        try:
            # Create a BytesIO object from the bytes
            pdf_stream = BytesIO(pdf_bytes)
            
            # Create an InputDocument
            in_doc = InputDocument(
                path_or_stream=pdf_stream,
                format=InputFormat.PDF,
                filename="uploaded.pdf",
            )
            
            # Convert PDF to markdown using docling
            result = self.converter.convert(in_doc)
            
            # Export to markdown
            markdown_text = result.document.export_to_markdown()
            
            logger.info("Successfully processed PDF bytes")
            return markdown_text
            
        except Exception as e:
            logger.error(f"Error processing PDF bytes: {e}")
            raise Exception(f"Error processing PDF bytes: {str(e)}")
