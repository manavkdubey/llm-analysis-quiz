"""Data processing utilities for various file formats."""
import pandas as pd
import numpy as np
import pdfplumber
import PyPDF2
import requests
import io
import base64
from typing import Optional, Union, Dict, Any
import logging
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)


def download_file(url: str, headers: Optional[Dict[str, str]] = None) -> bytes:
    """
    Download a file from URL.
    
    Args:
        url: URL to download from
        headers: Optional HTTP headers
    
    Returns:
        File content as bytes
    """
    logger.info(f"Downloading file from {url}")
    response = requests.get(url, headers=headers or {}, timeout=60, allow_redirects=True)
    response.raise_for_status()
    return response.content


def process_pdf(file_path: str, page: Optional[int] = None) -> Union[str, pd.DataFrame]:
    """
    Process PDF file and extract text or tables.
    
    Args:
        file_path: Path to PDF file
        page: Specific page number (1-indexed), or None for all pages
    
    Returns:
        Extracted text or DataFrame with tables
    """
    logger.info(f"Processing PDF: {file_path}, page: {page}")
    
    # Try pdfplumber first (better for tables)
    try:
        with pdfplumber.open(file_path) as pdf:
            if page:
                pages = [pdf.pages[page - 1]]
            else:
                pages = pdf.pages
            
            text_parts = []
            tables = []
            
            for p in pages:
                text = p.extract_text()
                if text:
                    text_parts.append(text)
                
                # Extract tables
                page_tables = p.extract_tables()
                if page_tables:
                    for table in page_tables:
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df)
            
            if tables:
                # Return the first table or concatenated tables
                if len(tables) == 1:
                    return tables[0]
                else:
                    return pd.concat(tables, ignore_index=True)
            else:
                return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")
    
    # Fallback to PyPDF2
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            if page:
                pages = [pdf_reader.pages[page - 1]]
            else:
                pages = pdf_reader.pages
            
            text_parts = []
            for p in pages:
                text = p.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")
        raise


def process_csv(content: Union[str, bytes], **kwargs) -> pd.DataFrame:
    """
    Process CSV content into DataFrame.
    
    Args:
        content: CSV content as string or bytes
        **kwargs: Additional arguments for pd.read_csv
    
    Returns:
        DataFrame
    """
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    
    return pd.read_csv(io.StringIO(content), **kwargs)


def process_excel(content: bytes, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Process Excel file into DataFrame.
    
    Args:
        content: Excel file content as bytes
        sheet_name: Specific sheet name, or None for first sheet
    
    Returns:
        DataFrame
    """
    return pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)


def analyze_data(df: pd.DataFrame, operation: str, **kwargs) -> Any:
    """
    Perform data analysis operations.
    
    Args:
        df: DataFrame to analyze
        operation: Operation to perform (sum, mean, count, filter, etc.)
        **kwargs: Additional arguments for the operation
    
    Returns:
        Result of the operation
    """
    logger.info(f"Performing {operation} on data")
    
    if operation == "sum":
        column = kwargs.get("column")
        if column:
            return float(df[column].sum())
        return float(df.select_dtypes(include=[np.number]).sum().sum())
    
    elif operation == "mean":
        column = kwargs.get("column")
        if column:
            return float(df[column].mean())
        return float(df.select_dtypes(include=[np.number]).mean().mean())
    
    elif operation == "count":
        return int(len(df))
    
    elif operation == "filter":
        # kwargs should contain filter conditions
        filtered = df.copy()
        for key, value in kwargs.items():
            if key != "operation":
                filtered = filtered[filtered[key] == value]
        return filtered
    
    elif operation == "groupby":
        by = kwargs.get("by")
        agg = kwargs.get("agg", "sum")
        if by:
            return df.groupby(by).agg(agg)
        return df
    
    else:
        raise ValueError(f"Unknown operation: {operation}")


def create_visualization(
    data: pd.DataFrame,
    viz_type: str,
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Create a visualization and return as base64 encoded image.
    
    Args:
        data: DataFrame to visualize
        viz_type: Type of visualization (bar, line, scatter, etc.)
        output_path: Optional path to save image
        **kwargs: Additional arguments for visualization
    
    Returns:
        Base64 encoded image string
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
    
    if viz_type == "bar":
        column = kwargs.get("column")
        if column:
            data[column].value_counts().plot(kind='bar', ax=ax)
        else:
            data.plot(kind='bar', ax=ax)
    
    elif viz_type == "line":
        data.plot(kind='line', ax=ax)
    
    elif viz_type == "scatter":
        x = kwargs.get("x")
        y = kwargs.get("y")
        if x and y:
            ax.scatter(data[x], data[y])
        else:
            data.plot(kind='scatter', ax=ax, x=data.columns[0], y=data.columns[1])
    
    elif viz_type == "hist":
        column = kwargs.get("column")
        if column:
            data[column].hist(ax=ax)
        else:
            data.hist(ax=ax)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

