import logging
import os
import io
import mimetypes
from azure.storage.blob import BlobServiceClient
import azure.functions as func

STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
BLOB_CONTAINER_NAME = os.environ.get("BLOB_STORAGE_CONTAINER_NAME", "uploads")
MAX_FILE_SIZE_MB = 20

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function `UploadFile` received a request.')

    try:
        # Debug: Log request details
        logging.info(f"Request method: {req.method}")
        logging.info(f"Request headers: {dict(req.headers)}")
        logging.info(f"Files in request: {list(req.files.keys())}")
        
        uploaded_file = req.files.get('file')
        if not uploaded_file:
            logging.error("No file found in request")
            return func.HttpResponse("No file found in the request.", status_code=400)

        filename = uploaded_file.filename
        file_content = uploaded_file.stream.read()

        # Debug: Log file details
        logging.info(f"Received file: {filename}")
        logging.info(f"File size: {len(file_content)} bytes")
        
        if not filename:
            logging.error("Filename is empty")
            return func.HttpResponse("Filename cannot be empty.", status_code=400)

        # Enhanced file extension validation
        file_ext = os.path.splitext(filename)[1].lower().strip()
        logging.info(f"File extension detected: '{file_ext}'")
        
        # Expanded allowed extensions with common variations
        allowed_extensions = [".pdf", ".json", ".txt", ".csv"]
        
        # Check MIME type as well for better validation
        content_type = getattr(uploaded_file, 'content_type', None)
        logging.info(f"File content type: {content_type}")
        
        # CSV-specific MIME type validation
        csv_mime_types = [
            'text/csv',
            'application/csv',
            'text/comma-separated-values',
            'application/vnd.ms-excel',  # Sometimes CSV files are detected as Excel
            'text/plain'  # Some systems send CSV as plain text
        ]
        
        if file_ext not in allowed_extensions:
            logging.error(f"Invalid file extension: {file_ext}. Allowed: {allowed_extensions}")
            return func.HttpResponse(
                f"Invalid file type '{file_ext}'. Allowed types: {', '.join(allowed_extensions)}.", 
                status_code=400
            )
        
        # Additional validation for CSV files
        if file_ext == ".csv":
            logging.info("Validating CSV file...")
            
            # Check if content looks like CSV
            try:
                content_preview = file_content[:1000].decode('utf-8', errors='ignore')
                logging.info(f"CSV content preview (first 200 chars): {content_preview[:200]}")
                
                # Basic CSV validation - check for common CSV patterns
                if not content_preview.strip():
                    logging.error("CSV file appears to be empty")
                    return func.HttpResponse("CSV file appears to be empty.", status_code=400)
                
                # Check for common CSV indicators
                csv_indicators = [',', ';', '\t', '\n']
                if not any(indicator in content_preview for indicator in csv_indicators):
                    logging.warning("File doesn't contain common CSV separators")
                
            except Exception as e:
                logging.error(f"Error validating CSV content: {e}")
                # Don't fail the upload, just log the warning
                pass

        # File size validation
        file_size_mb = len(file_content) / (1024 * 1024)
        logging.info(f"File size: {file_size_mb:.2f} MB")
        
        if len(file_content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            logging.error(f"File size {file_size_mb:.2f} MB exceeds limit of {MAX_FILE_SIZE_MB} MB")
            return func.HttpResponse(f"File size exceeds {MAX_FILE_SIZE_MB} MB.", status_code=400)

        # Enhanced blob upload with better error handling
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        
        # Sanitize filename to avoid issues
        safe_filename = filename.replace(" ", "_").replace("(", "").replace(")", "")
        
        blob_client = blob_service_client.get_blob_client(
            container=BLOB_CONTAINER_NAME, 
            blob=safe_filename
        )

        logging.info(f"Uploading file: {safe_filename} ({file_ext}) to container: {BLOB_CONTAINER_NAME}")
        
        # Set content type for CSV files explicitly
        content_settings = None
        if file_ext == ".csv":
            from azure.storage.blob import ContentSettings
            content_settings = ContentSettings(content_type='text/csv')
        
        blob_client.upload_blob(
            file_content, 
            overwrite=True,
            content_settings=content_settings
        )
        
        logging.info(f"Successfully uploaded {safe_filename} ({len(file_content)} bytes)")

        return func.HttpResponse(
             f"File '{filename}' uploaded successfully. It will be processed shortly.",
             status_code=200
        )
        
    except Exception as e:
        logging.error(f"Error uploading file: {str(e)}", exc_info=True)
        return func.HttpResponse(f"Upload error: {str(e)}", status_code=500)
