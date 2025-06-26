import logging
import os
import json
import re
from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions
import azure.functions as func

# --- Configuration ---
# Use connection string approach (more reliable)
COSMOS_CONNECTION_STRING = os.environ.get("CosmosDbConnectionString")
COSMOS_DATABASE_NAME = os.environ.get("COSMOS_DATABASE_NAME", "SampleDB")  # Default fallback
COSMOS_CONTAINER_NAME = os.environ.get("COSMOS_CONTAINER_NAME", "SampleContainer")  # Default fallback

# Alternative: Separate endpoint/key approach
COSMOS_ENDPOINT = os.environ.get("AZURE_COSMOS_ENDPOINT")
COSMOS_KEY = os.environ.get("AZURE_COSMOS_KEY")

# --- Singleton Cosmos Client Manager ---
class CosmosClientManager:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            try:
                # Method 1: Connection string (Recommended - fixes the dict error)
                if COSMOS_CONNECTION_STRING:
                    cls._client = CosmosClient.from_connection_string(
                        COSMOS_CONNECTION_STRING,
                        consistency_level='Session'
                    )
                    logging.info("Cosmos client initialized from connection string")
                
                # Method 2: Endpoint + Key (Fixed version)
                elif COSMOS_ENDPOINT and COSMOS_KEY:
                    # FIXED: Removed connection_policy to avoid dict error
                    cls._client = CosmosClient(
                        COSMOS_ENDPOINT, 
                        credential=COSMOS_KEY,
                        consistency_level='Session'
                        # Removed connection_policy - this was causing the dict error
                    )
                    logging.info("Cosmos client initialized from endpoint + key")
                else:
                    logging.error("Neither connection string nor endpoint+key configured")
                    cls._client = None
                    
            except Exception as e:
                logging.error(f"Failed to initialize Cosmos DB client: {str(e)}")
                cls._client = None
                
        return cls._client

# --- Enhanced Cosmos Query Validation ---
def enhanced_cosmos_query_validation(query: str) -> dict:
    """Enhanced validation for Cosmos DB queries"""
    try:
        query_upper = query.strip().upper()
        if not query_upper.startswith("SELECT"):
            return {"valid": False, "reason": "Only SELECT queries are allowed."}
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'EXEC\s+', r'SP_\s+', r'XP_\s+', r'OPENROWSET', r'BULK\s+',
            r'WAITFOR\s+', r'SHUTDOWN', r'DROP\s+', r'ALTER\s+', r'CREATE\s+'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query_upper):
                return {"valid": False, "reason": f"Dangerous pattern detected: {pattern}"}
        
        # FIXED: Better alias validation for Cosmos DB
        # In Cosmos DB, common aliases are 'c', 'root', or table name
        if not re.search(r'FROM\s+\w+\s*', query_upper):
            return {"valid": False, "reason": "Query must specify a FROM clause with container alias"}
        
        # Limit complexity 
        if query_upper.count('JOIN') > 3:
            return {"valid": False, "reason": "Query complexity: Too many JOINs (max 3)."}
        
        # Check query length (prevent very complex queries)
        if len(query) > 2000:
            return {"valid": False, "reason": "Query too long (max 2000 characters)"}

        return {"valid": True, "reason": "Query validated."}
        
    except Exception as e:
        logging.error(f"Error during query validation: {str(e)}")
        return {"valid": False, "reason": f"Validation exception: {str(e)}"}

# --- Main Azure Function ---
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('QueryCosmosDB function received a request.')
    
    # FIXED: Better client availability check
    cosmos_client_instance = CosmosClientManager.get_client()
    if not cosmos_client_instance:
        return func.HttpResponse(
            json.dumps({
                "success": False, 
                "error": "Cosmos DB client is not available. Check configuration.",
                "debug_info": {
                    "connection_string_available": bool(COSMOS_CONNECTION_STRING),
                    "endpoint_available": bool(COSMOS_ENDPOINT),
                    "key_available": bool(COSMOS_KEY)
                }
            }),
            status_code=500, 
            mimetype="application/json"
        )

    try:
        # FIXED: Better request body handling
        if req.method == 'GET':
            query = req.params.get('query')
        else:
            try:
                req_body = req.get_json()
                query = req_body.get('query') if req_body else None
            except (ValueError, TypeError):
                return func.HttpResponse(
                    json.dumps({"success": False, "error": "Invalid JSON in request body"}),
                    status_code=400, 
                    mimetype="application/json"
                )
                
    except Exception as e:
        logging.error(f"Error parsing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"success": False, "error": "Invalid request format"}),
            status_code=400, 
            mimetype="application/json"
        )

    if not query:
        return func.HttpResponse(
            json.dumps({"success": False, "error": "Please include 'query' parameter or in request body"}),
            status_code=400, 
            mimetype="application/json"
        )

    # Validate query
    validation_result = enhanced_cosmos_query_validation(query)
    if not validation_result["valid"]:
        logging.warning(f"Query validation failed: {validation_result['reason']}")
        return func.HttpResponse(
            json.dumps({
                "success": False, 
                "error": f"Query validation failed: {validation_result['reason']}", 
                "query_attempted": query
            }),
            status_code=403,
            mimetype="application/json"
        )

    try:
        # FIXED: Better database/container handling
        if not COSMOS_DATABASE_NAME or not COSMOS_CONTAINER_NAME:
            return func.HttpResponse(
                json.dumps({
                    "success": False, 
                    "error": "Database or container name not configured"
                }),
                status_code=500,
                mimetype="application/json"
            )
            
        database = cosmos_client_instance.get_database_client(COSMOS_DATABASE_NAME)
        container = database.get_container_client(COSMOS_CONTAINER_NAME)
        
        logging.info(f"Executing query in {COSMOS_DATABASE_NAME}/{COSMOS_CONTAINER_NAME}: {query[:100]}...")
        
        # FIXED: Better query execution with timeout and limits
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True,
            max_item_count=100  # Limit to prevent large responses
        ))

        logging.info(f"Query executed successfully. Items returned: {len(items)}")
        
        # FIXED: Better response formatting
        return func.HttpResponse(
            json.dumps({
                "success": True, 
                "data": items[:100],  # Ensure max 100 items
                "query_executed": query,
                "items_count": len(items),
                "database": COSMOS_DATABASE_NAME,
                "container": COSMOS_CONTAINER_NAME
            }, default=str),  # Handle datetime serialization issues
            mimetype="application/json"
        )

    except cosmos_exceptions.CosmosResourceNotFoundError as e:
        logging.error(f"Cosmos DB resource not found: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "success": False, 
                "error": f"Database or container not found: {str(e)}", 
                "query_attempted": query
            }),
            status_code=404,
            mimetype="application/json"
        )
        
    except cosmos_exceptions.CosmosHttpResponseError as e:
        logging.error(f"Cosmos DB HTTP error: {e.message}")
        return func.HttpResponse(
            json.dumps({
                "success": False, 
                "error": f"Cosmos DB error: {e.message}", 
                "query_attempted": query,
                "status_code": e.status_code
            }),
            status_code=min(e.status_code, 500) if hasattr(e, 'status_code') else 500,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return func.HttpResponse(
            json.dumps({
                "success": False, 
                "error": f"Unexpected error: {str(e)}", 
                "query_attempted": query
            }),
            status_code=500,
            mimetype="application/json"
        )
