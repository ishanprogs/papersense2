import logging
import os
import io
import uuid
import base64
import json
import fitz  # PyMuPDF (fallback)
from datetime import datetime
import re
import csv
import pandas as pd

# Document Intelligence Imports
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential as AzureCoreKeyCredential

# Azure Search, OpenAI, Langchain, Functions imports
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, SimpleField,
    SearchableField, VectorSearch, HnswAlgorithmConfiguration,
    VectorSearchProfile
)
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import azure.functions as func

# Configuration
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "pdf-index")

OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
VECTOR_DIMENSION = 3072

# Document Intelligence Configuration
DOC_INTEL_ENDPOINT = os.environ.get("DOC_INTEL_ENDPOINT")
DOC_INTEL_KEY = os.environ.get("DOC_INTEL_KEY")

# Validate required environment variables
required_env_vars = [
    "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY", "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
]
for var in required_env_vars:
    if not os.environ.get(var):
        logging.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")

# Initialize clients
search_index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_KEY))
openai_client = AzureOpenAI(
    api_key=OPENAI_KEY,
    azure_endpoint=OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION
)

# Initialize Document Intelligence Client
doc_intelligence_client = None
if DOC_INTEL_ENDPOINT and DOC_INTEL_KEY:
    try:
        doc_intelligence_client = DocumentIntelligenceClient(
            endpoint=DOC_INTEL_ENDPOINT,
            credential=AzureCoreKeyCredential(DOC_INTEL_KEY)
        )
        logging.info("Document Intelligence client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Document Intelligence client: {str(e)}")
        doc_intelligence_client = None
else:
    logging.warning("Document Intelligence credentials not configured.")

def create_search_index_if_not_exists():
    try:
        search_index_client.get_index(SEARCH_INDEX_NAME)
        logging.info(f"Search index '{SEARCH_INDEX_NAME}' already exists.")
    except Exception:
        logging.info(f"Search index '{SEARCH_INDEX_NAME}' not found. Creating new index...")
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, sortable=False, filterable=False, facetable=False),
            SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=VECTOR_DIMENSION, vector_search_profile_name="my-hnsw-profile"),
            SimpleField(name="source_file", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SimpleField(name="file_type", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
            SearchableField(name="document_type", type=SearchFieldDataType.String, default_value="document", sortable=True, filterable=True, facetable=True),
            SearchableField(name="json_path", type=SearchFieldDataType.String, default_value="N/A", sortable=True, filterable=True, facetable=False),
            SimpleField(name="chunk_seq", type=SearchFieldDataType.Int32, sortable=True, filterable=True, facetable=False)
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="my-hnsw-config")],
            profiles=[VectorSearchProfile(name="my-hnsw-profile", algorithm_configuration_name="my-hnsw-config")]
        )
        index = SearchIndex(name=SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search)
        search_index_client.create_index(index)
        logging.info(f"Search index '{SEARCH_INDEX_NAME}' created.")

def get_embeddings(text_or_texts):
    if isinstance(text_or_texts, str): 
        text_or_texts = [text_or_texts]
    if not text_or_texts: 
        return []
    valid_texts = [t for t in text_or_texts if t and t.strip()]
    if not valid_texts: 
        return []
    response = openai_client.embeddings.create(input=valid_texts, model=OPENAI_EMBEDDING_DEPLOYMENT)
    return [item.embedding for item in response.data]

def detect_document_type(content_text):
    """Detect document type based on content analysis"""
    content_lower = content_text.lower()
    
    # Invoice/Receipt indicators
    invoice_keywords = ['invoice', 'bill', 'receipt', 'total amount', 'tax id', 'invoice number', 
                       'billing address', 'invoice date', 'due date', 'amount due', 'subtotal']
    
    # Service agreement indicators  
    agreement_keywords = ['agreement', 'contract', 'terms and conditions', 'service agreement',
                         'vehicle service agreement', 'warranty', 'coverage', 'obligor', 'agreement holder']
    
    # Form indicators
    form_keywords = ['application', 'form', 'please complete', 'signature', 'date signed']
    
    invoice_score = sum(1 for keyword in invoice_keywords if keyword in content_lower)
    agreement_score = sum(1 for keyword in agreement_keywords if keyword in content_lower)
    form_score = sum(1 for keyword in form_keywords if keyword in content_lower)
    
    if invoice_score >= 3:
        return "invoice"
    elif agreement_score >= 3:
        return "agreement"
    elif form_score >= 2:
        return "form"
    else:
        return "document"

def extract_text_with_document_intelligence(file_bytes, filename):
    """Enhanced Document Intelligence extraction with correct API calls"""
    if not doc_intelligence_client:
        logging.error("Document Intelligence client not available")
        return None, "document"
    
    try:
        # Convert bytes to BytesIO stream for better compatibility
        import io
        file_stream = io.BytesIO(file_bytes)
        
        # First pass: Use layout model to get basic content
        logging.info(f"Analyzing document structure for {filename}")
        layout_poller = doc_intelligence_client.begin_analyze_document(
            "prebuilt-layout", 
            body=file_stream,  # Correct usage with body parameter
            features=["styleFont"]  # Simplified features
        )
        layout_result = layout_poller.result()
        
        # Detect document type from content
        initial_content = layout_result.content if layout_result.content else ""
        doc_type = detect_document_type(initial_content)
        logging.info(f"Detected document type: {doc_type} for {filename}")
        
        # Reset stream position for second call
        file_stream.seek(0)
        
        # Choose appropriate model based on detected type
        if doc_type == "invoice":
            model_id = "prebuilt-invoice"
        else:
            model_id = "prebuilt-layout"  # Use layout for agreements, forms, and general documents
        
        # Second pass: Use optimal model for final extraction
        logging.info(f"Processing {filename} with model: {model_id}")
        poller = doc_intelligence_client.begin_analyze_document(
            model_id,
            body=file_stream,  # Correct usage with body parameter
            features=["styleFont"]  # Simplified features for compatibility
        )
        result = poller.result()
        
        # Build comprehensive extracted text
        extracted_text = build_structured_content(result, doc_type, filename)
        
        return extracted_text, doc_type
        
    except Exception as e:
        logging.error(f"Document Intelligence extraction failed for {filename}: {str(e)}")
        return None, "document"

def build_structured_content(result, doc_type, filename):
    """Build well-structured content from Document Intelligence results"""
    content_parts = []
    
    # Add document header with metadata
    content_parts.append(f"=== DOCUMENT: {filename} ===")
    content_parts.append(f"Document Type: {doc_type.upper()}")
    content_parts.append(f"Processed Date: {datetime.utcnow().isoformat()}")
    content_parts.append("")
    
    # Add main content with better formatting
    if hasattr(result, 'content') and result.content:
        content_parts.append("=== MAIN CONTENT ===")
        # Clean and format main content
        cleaned_content = clean_and_format_text(result.content)
        content_parts.append(cleaned_content)
        content_parts.append("")
    
    # Extract and format key-value pairs for forms and agreements
    if hasattr(result, 'key_value_pairs') and result.key_value_pairs:
        content_parts.append("=== FORM FIELDS & KEY INFORMATION ===")
        kv_pairs = []
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                key_text = clean_text(kv_pair.key.content) if kv_pair.key.content else "Unknown Field"
                value_text = clean_text(kv_pair.value.content) if kv_pair.value.content else "Not Specified"
                
                # Add confidence if available
                confidence_info = ""
                if hasattr(kv_pair, 'confidence') and kv_pair.confidence:
                    confidence_info = f" (Confidence: {kv_pair.confidence:.2f})"
                
                kv_pairs.append(f"• {key_text}: {value_text}{confidence_info}")
        
        if kv_pairs:
            content_parts.extend(kv_pairs)
            content_parts.append("")
        logging.info(f"Extracted {len(kv_pairs)} key-value pairs from {filename}")
    
    # Extract and format tables with better structure
    if hasattr(result, 'tables') and result.tables:
        content_parts.append("=== TABLES & STRUCTURED DATA ===")
        for i, table in enumerate(result.tables):
            content_parts.append(f"--- Table {i+1} ({table.row_count} rows x {table.column_count} columns) ---")
            
            # Build table in markdown format for better readability
            table_content = build_markdown_table(table)
            content_parts.append(table_content)
            content_parts.append("")
        
        logging.info(f"Extracted {len(result.tables)} tables from {filename}")
    
    # Add style and formatting information
    if hasattr(result, 'styles') and result.styles:
        style_info = []
        for style in result.styles:
            if hasattr(style, 'font_weight') and style.font_weight == "bold":
                style_info.append("Contains bold text formatting")
            if hasattr(style, 'font_style') and style.font_style == "italic":
                style_info.append("Contains italic text formatting")
        
        if style_info:
            content_parts.append("=== DOCUMENT FORMATTING ===")
            content_parts.extend([f"• {info}" for info in style_info])
            content_parts.append("")
    
    # Add sections detection for agreements and contracts
    if doc_type in ["agreement", "contract"]:
        sections = detect_document_sections(result.content if result.content else "")
        if sections:
            content_parts.append("=== DOCUMENT SECTIONS ===")
            content_parts.extend([f"• {section}" for section in sections])
            content_parts.append("")
    
    return "\n".join(content_parts)

def build_markdown_table(table):
    """Convert Document Intelligence table to markdown format"""
    try:
        # Create a matrix to hold cell contents
        table_matrix = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
        
        # Fill the matrix with cell contents
        for cell in table.cells:
            if cell.row_index < table.row_count and cell.column_index < table.column_count:
                content = clean_text(cell.content) if cell.content else ""
                table_matrix[cell.row_index][cell.column_index] = content
        
        # Convert to markdown
        markdown_lines = []
        
        # Header row
        if table.row_count > 0:
            header = "| " + " | ".join(table_matrix[0]) + " |"
            markdown_lines.append(header)
            
            # Separator row
            separator = "| " + " | ".join(["---"] * table.column_count) + " |"
            markdown_lines.append(separator)
            
            # Data rows
            for row_idx in range(1, table.row_count):
                row = "| " + " | ".join(table_matrix[row_idx]) + " |"
                markdown_lines.append(row)
        
        return "\n".join(markdown_lines)
    
    except Exception as e:
        logging.error(f"Error building markdown table: {str(e)}")
        # Fallback to simple format
        simple_table = []
        for cell in table.cells:
            simple_table.append(f"Row {cell.row_index}, Col {cell.column_index}: {cell.content}")
        return "\n".join(simple_table)

def clean_text(text):
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # FIXED: Properly escaped regex pattern
    text = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]\{\}"\'/@#$%&*+=<>]', ' ', text)
    
    return text

def clean_and_format_text(content):
    """Enhanced text cleaning and formatting"""
    if not content:
        return ""
    
    # Normalize line breaks and spaces
    content = re.sub(r'\r\n|\r|\n', '\n', content)
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Multiple line breaks to double
    content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces to single
    
    # Preserve important formatting markers
    content = re.sub(r'\n([A-Z][A-Z\s]+:)', r'\n\n\1', content)  # Section headers
    content = re.sub(r'\n(\d+\.)', r'\n\1', content)  # Numbered lists
    
    return content.strip()

def detect_document_sections(content):
    """Detect major sections in agreements and contracts"""
    if not content:
        return []
    
    sections = []
    # Common section patterns in agreements
    section_patterns = [
        r'I+\.\s*([A-Z][A-Z\s]+)',  # Roman numerals
        r'SECTION\s+\d+[:\.]?\s*([A-Z][A-Z\s]+)',  # Section numbers
        r'^([A-Z][A-Z\s]{10,}):?$',  # All caps headers
        r'ARTICLE\s+\d+[:\.]?\s*([A-Z][A-Z\s]+)',  # Article numbers
    ]
    
    for pattern in section_patterns:
        matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
        sections.extend([match.strip() for match in matches if len(match.strip()) > 3])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sections = []
    for section in sections:
        if section not in seen:
            seen.add(section)
            unique_sections.append(section)
    
    return unique_sections[:10]  # Limit to top 10 sections

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Enhanced chunking that respects document structure"""
    if not text or not text.strip():
        return []
    
    # Custom separators that respect document structure
    separators = [
        "\n=== ",  # Section breaks
        "\n--- ",  # Sub-section breaks
        "\n\n\n",  # Multiple line breaks
        "\n\n",    # Paragraph breaks
        "\n• ",    # List items
        "\n",      # Line breaks
        ". ",      # Sentences
        ", ",      # Clauses
        " "        # Words
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=separators,
        keep_separator=True
    )
    
    chunks = text_splitter.split_text(text)
    
    # Post-process chunks to ensure they have meaningful content
    meaningful_chunks = []
    for chunk in chunks:
        if len(chunk.strip()) > 50:  # Only keep substantial chunks
            meaningful_chunks.append(chunk.strip())
    
    return meaningful_chunks

def json_to_descriptive_text_enhanced(data, config=None):
    """Enhanced JSON processing"""
    if config is None:
        config = {}
    if isinstance(data, dict):
        return json.dumps(data, indent=2)
    elif isinstance(data, list):
        return "\n".join([json.dumps(item, indent=2) for item in data])
    return str(data)

def process_csv_content(file_bytes, filename):
    """Process CSV file content into searchable format"""
    try:
        # Decode CSV content
        csv_content = file_bytes.decode('utf-8')
        csv_reader = csv.reader(csv_content.splitlines())
        
        # Convert to list for processing
        rows = list(csv_reader)
        if not rows:
            return ""
        
        # Get headers (first row)
        headers = rows[0]
        data_rows = rows[1:]
        
        content_parts = []
        
        # Add document header
        content_parts.append(f"=== CSV DOCUMENT: {filename} ===")
        content_parts.append(f"Document Type: CSV DATA TABLE")
        content_parts.append(f"Processed Date: {datetime.utcnow().isoformat()}")
        content_parts.append(f"Total Rows: {len(data_rows)}")
        content_parts.append(f"Total Columns: {len(headers)}")
        content_parts.append("")
        
        # Add column information
        content_parts.append("=== COLUMN INFORMATION ===")
        # Ensure headers is a list of strings
        if all(isinstance(h, str) for h in headers):
            for i, header in enumerate(headers):
                content_parts.append(f"Column {i+1}: {header}")
        else:
            logging.warning(f"CSV header row is not a list of strings in {filename}: {headers}")
            for i, header in enumerate(headers):
                content_parts.append(f"Column {i+1}: {str(header)}")
        content_parts.append("")
        
        # Add structured data in multiple formats for better searchability
        
        # Format 1: Row-by-row with field names
        content_parts.append("=== STRUCTURED DATA RECORDS ===")
        for row_idx, row in enumerate(data_rows[:100]):  # Limit to first 100 rows for large CSVs
            content_parts.append(f"--- Record {row_idx + 1} ---")
            for col_idx, value in enumerate(row):
                if col_idx < len(headers):
                    column_name = headers[col_idx]
                    content_parts.append(f"{column_name}: {value}")
            content_parts.append("")
        
        # Format 2: Summary statistics for numeric columns
        content_parts.append("=== DATA SUMMARY ===")
        try:
            df = pd.DataFrame(data_rows, columns=headers)
            
            # Identify numeric columns
            numeric_columns = []
            for col in headers:
                try:
                    pd.to_numeric(df[col], errors='raise')
                    numeric_columns.append(col)
                except:
                    pass
            
            if numeric_columns:
                content_parts.append("Numeric Columns Summary:")
                for col in numeric_columns[:5]:  # Limit to first 5 numeric columns
                    try:
                        values = pd.to_numeric(df[col], errors='coerce').dropna()
                        if not values.empty:
                            content_parts.append(f"• {col}: Min={values.min()}, Max={values.max()}, Average={values.mean():.2f}")
                    except:
                        pass
                content_parts.append("")
            
            # Add categorical data summary
            categorical_columns = [col for col in headers if col not in numeric_columns]
            if categorical_columns:
                content_parts.append("Categorical Columns Summary:")
                for col in categorical_columns[:5]:  # Limit to first 5 categorical columns
                    try:
                        unique_values = df[col].unique()[:10]  # First 10 unique values
                        content_parts.append(f"• {col}: {len(df[col].unique())} unique values, examples: {', '.join(map(str, unique_values))}")
                    except:
                        pass
                content_parts.append("")
                
        except Exception as e:
            logging.warning(f"Could not generate summary statistics for {filename}: {e}")
        
        # Format 3: Searchable key-value pairs for common queries
        content_parts.append("=== SEARCHABLE KEY-VALUE PAIRS ===")
        for row_idx, row in enumerate(data_rows[:50]):  # Limit for performance
            for col_idx, value in enumerate(row):
                if col_idx < len(headers) and value.strip():
                    column_name = headers[col_idx]
                    # Create searchable patterns
                    content_parts.append(f"• {column_name} contains {value}")
                    content_parts.append(f"• Record {row_idx + 1} has {column_name} = {value}")
        
        return "\n".join(content_parts)
        
    except Exception as e:
        logging.error(f"Error processing CSV content for {filename}: {str(e)}")
        return f"Error processing CSV file: {filename}"

def main(myblob: func.InputStream) -> None:
    logging.info(f"Python Blob trigger function `ProcessFile` processed blob: {myblob.name}")
    blob_filename_with_path = myblob.name
    blob_filename = os.path.basename(blob_filename_with_path)
    file_ext = os.path.splitext(blob_filename)[1].lower()

    documents_for_search_index = []

    try:
        create_search_index_if_not_exists()
        search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))
        file_bytes = myblob.read()
        
        primary_text_content = ""
        file_type_for_index = "unknown"
        document_type = "document"
        json_path_for_index = "N/A"

        if file_ext == ".pdf":
            file_type_for_index = "pdf"
            
            if doc_intelligence_client:
                extracted_text, doc_type = extract_text_with_document_intelligence(file_bytes, blob_filename)
                if extracted_text:
                    primary_text_content = extracted_text
                    document_type = doc_type
                    logging.info(f"Successfully extracted content using Document Intelligence for {blob_filename} (Type: {doc_type}, Length: {len(primary_text_content)})")
                else:
                    logging.warning(f"Document Intelligence failed, falling back to PyMuPDF for {blob_filename}")
                    # Fallback to PyMuPDF
                    try:
                        doc = fitz.open(stream=file_bytes, filetype="pdf")
                        for page in doc:
                            primary_text_content += page.get_text("text") + "\n"
                        doc.close()
                        document_type = "document"
                        logging.info(f"Fallback extraction completed for {blob_filename}")
                    except Exception as e_pdf:
                        logging.error(f"PyMuPDF fallback also failed for {blob_filename}: {str(e_pdf)}")
                        return
            else:
                logging.warning(f"Document Intelligence not configured, using PyMuPDF for {blob_filename}")
                try:
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    for page in doc:
                        primary_text_content += page.get_text("text") + "\n"
                    doc.close()
                    document_type = "document"
                except Exception as e_pdf:
                    logging.error(f"Error processing PDF {blob_filename}: {str(e_pdf)}")
                    return

        elif file_ext == ".json":
            file_type_for_index = "json"
            try:
                json_string = file_bytes.decode('utf-8')
                json_data = json.loads(json_string)
                meta_header = f"Content from JSON document: {blob_filename}. Generated on: {datetime.utcnow().isoformat()}.\n"
                descriptive_text = json_to_descriptive_text_enhanced(json_data)
                primary_text_content = meta_header + descriptive_text
                document_type = "json"
                logging.info(f"Converted JSON to descriptive text: {blob_filename} (Length: {len(primary_text_content)})")
            except Exception as e_json:
                logging.error(f"Error processing JSON {blob_filename}: {str(e_json)}")
                return
        elif file_ext == ".csv":
            file_type_for_index = "csv"
            try:
                primary_text_content = process_csv_content(file_bytes, blob_filename)
                document_type = "csv_data"
                logging.info(f"Processed CSV file: {blob_filename} (Length: {len(primary_text_content)})")
            except Exception as e_csv:
                logging.error(f"Error processing CSV {blob_filename}: {str(e_csv)}")
                return
        else:
            logging.warning(f"Unsupported file type: {file_ext} for blob: {blob_filename}. Skipping.")
            return

        if not primary_text_content or not primary_text_content.strip():
            logging.warning(f"No text content extracted for {blob_filename}. Nothing to index.")
            return

        # Enhanced chunking
        chunks = split_text_into_chunks(primary_text_content, chunk_size=1000, chunk_overlap=200)
        logging.info(f"Split '{blob_filename}' ({file_type_for_index}) into {len(chunks)} chunks.")

        if chunks:
            embeddings = get_embeddings(chunks)
            if len(embeddings) != len(chunks):
                logging.error(f"Mismatch in number of chunks and embeddings for {blob_filename}.")
                return

            for i, chunk_text in enumerate(chunks):
                encoded_filename_part = base64.urlsafe_b64encode(blob_filename.encode('utf-8')).decode('utf-8').rstrip('=')
                doc_key = f"{file_type_for_index}-{encoded_filename_part}-chunk{i}-{str(uuid.uuid4())[:8]}"
                current_json_path = json_path_for_index if file_type_for_index == "json" else "N/A"

                documents_for_search_index.append({
                    "id": doc_key,
                    "content": chunk_text,
                    "content_vector": embeddings[i],
                    "source_file": blob_filename,
                    "file_type": file_type_for_index,
                    "document_type": document_type,
                    "json_path": current_json_path,
                    "chunk_seq": i
                })
        
        if documents_for_search_index:
            search_client.upload_documents(documents=documents_for_search_index)
            logging.info(f"Successfully uploaded {len(documents_for_search_index)} documents for {blob_filename} (Type: {document_type}) to index '{SEARCH_INDEX_NAME}'.")
        else:
            logging.info(f"No documents were generated from {blob_filename} to upload.")

    except Exception as e:
        logging.error(f"General error processing blob {myblob.name}: {str(e)}", exc_info=True)

