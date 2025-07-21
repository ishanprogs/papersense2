import logging
import os
import json
import re
from datetime import datetime # For metadata in prompts
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
import azure.functions as func
import requests # For calling other Azure Functions

# === Configuration ===
# Azure Search
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "pdf-index")
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
OPENAI_CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
OPENAI_API_VERSION = "2024-10-21"

# Specific OpenAI deployments for specialized tasks
OPENAI_CHAT_DEPLOYMENT_FOR_NL2COSMOS = os.environ.get("OPENAI_CHAT_DEPLOYMENT_FOR_NL2COSMOS", OPENAI_CHAT_DEPLOYMENT)
OPENAI_CHAT_DEPLOYMENT_FOR_INTENT = os.environ.get("OPENAI_CHAT_DEPLOYMENT_FOR_INTENT", OPENAI_CHAT_DEPLOYMENT)

# Azure Cosmos DB (for NL Q&A)
COSMOS_ENDPOINT_FOR_PROMPT = os.environ.get("AZURE_COSMOS_ENDPOINT") # Used for schema description
COSMOS_DATABASE_NAME_FOR_PROMPT = os.environ.get("COSMOS_DATABASE_NAME", "AdventureWorksNoSQLDB")
COSMOS_CONTAINER_NAME_FOR_PROMPT = os.environ.get("COSMOS_CONTAINER_NAME", "AdventureWorksData")
QUERY_COSMOSDB_FUNCTION_URL = os.environ.get("QUERY_COSMOSDB_FUNCTION_URL") # URL of your QueryCosmosDB function
QUERY_COSMOSDB_FUNCTION_KEY = os.environ.get("QUERY_COSMOSDB_FUNCTION_KEY") # Key for QueryCosmosDB


# Initialize Clients
try:
    search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))
    openai_client = AzureOpenAI(
        api_key=OPENAI_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version=OPENAI_API_VERSION
    )
    logging.info("Azure Search and OpenAI clients initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize core Azure clients: {str(e)}")
    search_client = None
    openai_client = None

def get_embeddings(text):
    """Generates embeddings for the given text using Azure OpenAI."""
    response = openai_client.embeddings.create(
        input=text,
        model=OPENAI_EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding

def create_comprehensive_document_mapping():
    """Complete field mappings for maximum accuracy"""
    return {
        # Agreement Holder Information
        'agreement_holder_queries': {
            'patterns': [
                r'(?i).*(agreement holder|holder).*(name|who|identity).*',
                r'(?i).*who.*holder.*',
                r'(?i).*customer.*name.*',
                r'(?i).*obligor.*name.*',
                r'(?i).*subscriber.*name.*'
            ],
            'field_mappings': [
                "Agreement Holder", "Agreement Holder name", "Boris Singh",
                "customer name", "obligor", "subscriber", "holder name",
                "primary holder", "account holder", "policyholder"
            ],
            'section_context': ["AGREEMENT HOLDER INFORMATION", "Customer Information", "Subscriber Details"]
        },
        
        'address_queries': {
            'patterns': [
                r'(?i).*(agreement holder|holder|customer).*(address|location|where|live|residence).*',
                r'(?i).*where.*live.*',
                r'(?i).*mailing.*address.*',
                r'(?i).*home.*address.*'
            ],
            'field_mappings': [
                "Agreement Holder Address", "holder address", "customer address",
                "8818 Lew Wallace Rd", "Frederick", "Maryland", "21704",
                "mailing address", "residence address", "home address",
                "street address", "city", "state", "zip code", "postal code"
            ],
            'section_context': ["AGREEMENT HOLDER INFORMATION", "Address INFORMATION"]
        },
        
        'contact_queries': {
            'patterns': [
                r'(?i).*(agreement holder|holder|customer).*(phone|telephone|contact|number).*',
                r'(?i).*(home|business|work).*telephone.*',
                r'(?i).*contact.*(number|info).*',
                r'(?i).*phone.*number.*'
            ],
            'field_mappings': [
                "Home Telephone", "Business Telephone", "(301) 874-2",
                "phone number", "contact number", "telephone", "mobile",
                "work phone", "cell phone", "contact information"
            ],
            'section_context': ["AGREEMENT HOLDER INFORMATION", "Contact Information"]
        },
        
        # Vehicle Information
        'vehicle_queries': {
            'patterns': [
                r'(?i).*vehicle.*(info|detail|specification|data).*',
                r'(?i).*(car|automobile|auto).*(info|detail).*',
                r'(?i).*(make|model|year|vin).*(vehicle|car).*',
                r'(?i).*what.*(vehicle|car).*',
                r'(?i).*vehicle.*(covered|protected).*'
            ],
            'field_mappings': [
                "Vehicle Information", "2016 Toyota Land Cruiser", 
                "Year", "Make", "Model", "Vehicle Identification Number", 
                "VIN", "JTMCY7AJ8G4038164", "Current Odometer", "88798",
                "Vehicle Purchase Date", "01-01-2024", "mileage",
                "vehicle year", "vehicle make", "vehicle model"
            ],
            'section_context': ["VEHICLE INFORMATION", "Vehicle Details", "Auto Information"]
        },
        
        # Agreement/Coverage Details
        'agreement_queries': {
            'patterns': [
                r'(?i).*agreement.*(number|id|reference).*',
                r'(?i).*(policy|contract).*(number|id).*',
                r'(?i).*coverage.*(term|period|duration|length).*',
                r'(?i).*agreement.*(date|when|start|expire).*',
                r'(?i).*deductible.*',
                r'(?i).*premium.*cost.*'
            ],
            'field_mappings': [
                "Agreement Number", "VSC0000014", "Agreement Term", "39 months",
                "Agreement Purchase Date", "04-25-2024", "Agreement Expiration Date",
                "04-01-2027", "Deductible", "$300", "policy number",
                "contract number", "coverage period", "term length",
                "start date", "end date", "expiration", "renewal"
            ],
            'section_context': ["COVERAGE DESCRIPTION", "Agreement Details", "Policy Information"]
        },
        
        # Dealer Information
        'dealer_queries': {
            'patterns': [
                r'(?i).*(dealer|seller|dealership).*(name|info|contact|who).*',
                r'(?i).*where.*(bought|purchased|acquired).*',
                r'(?i).*purchase.*location.*',
                r'(?i).*sold.*by.*',
                r'(?i).*dealer.*contact.*'
            ],
            'field_mappings': [
                "Dealer/Seller", "Payton Car Lot", "6325 President George Bush",
                "Dallas", "Texas", "Dealer/Seller Number", "PAYTO-005",
                "Dealer/Seller Telephone", "(329) 875-1325", "dealership",
                "seller information", "dealer contact", "dealer address",
                "dealer phone", "dealer number"
            ],
            'section_context': ["DEALER/SELLER INFORMATION", "Dealer Details", "Seller Information"]
        },
        
        # Financial Information
        'financial_queries': {
            'patterns': [
                r'(?i).*(cost|price|premium|payment|fee).*',
                r'(?i).*how.*much.*',
                r'(?i).*(monthly|annual).*(payment|cost).*',
                r'(?i).*deductible.*amount.*'
            ],
            'field_mappings': [
                "premium", "cost", "payment", "fee", "deductible", "$300",
                "monthly payment", "annual cost", "total cost", "price"
            ],
            'section_context': ["COVERAGE DESCRIPTION", "Financial Information", "Cost Details"]
        },
        
        # CSV Data Queries
        'csv_data_queries': {
            'patterns': [
                r'(?i).*(data|records?|rows?|entries).*(contain|have|show|with).*',
                r'(?i).*(find|search|look for).*(records?|data|entries).*',
                r'(?i).*(how many|count|total).*(records?|rows?|entries).*',
                r'(?i).*(average|mean|sum|total|min|max|minimum|maximum).*',
                r'(?i).*(column|field).*(contains?|values?|data).*'
            ],
            'field_mappings': [
                "data records", "CSV data", "table data", "structured data",
                "columns", "rows", "fields", "entries", "records",
                "contains", "values", "statistics", "summary"
            ],
            'section_context': ["STRUCTURED DATA RECORDS", "DATA SUMMARY", "COLUMN INFORMATION"]
        },
        
        'data_analysis_queries': {
            'patterns': [
                r'(?i).*(analyze|analysis|statistics|stats).*',
                r'(?i).*(trend|pattern|distribution).*',
                r'(?i).*(compare|comparison|versus|vs).*',
                r'(?i).*(correlation|relationship|related).*'
            ],
            'field_mappings': [
                "data analysis", "statistics", "summary", "trends", "patterns",
                "numeric columns", "categorical columns", "unique values"
            ],
            'section_context': ["DATA SUMMARY", "SEARCHABLE KEY-VALUE PAIRS"]
        }
    }

def enhance_query_with_comprehensive_context(user_question):
    """Advanced query enhancement with comprehensive document understanding"""
    mappings = create_comprehensive_document_mapping()
    enhanced_terms = [user_question]
    matched_categories = []
    confidence_score = 0
    
    # Find all matching categories (not just first match)
    for category, config in mappings.items():
        category_confidence = 0
        for pattern in config['patterns']:
            if re.match(pattern, user_question):
                enhanced_terms.extend(config['field_mappings'])
                enhanced_terms.extend(config['section_context'])
                matched_categories.append(category)
                category_confidence += 1
                break
        
        # Also check for partial matches for broader coverage
        question_words = set(user_question.lower().split())
        field_words = set(' '.join(config['field_mappings']).lower().split())
        overlap = len(question_words.intersection(field_words))
        
        if overlap >= 2:  # Significant word overlap
            enhanced_terms.extend(config['field_mappings'][:8])  # Add top 8 terms
            category_confidence += 0.5
        
        confidence_score += category_confidence
    
    # Add document-wide context terms
    document_context_terms = [
        "Vehicle Service Agreement", "Schedule Page", "HWG", "Headstart Warranty Group",
        "Agreement Holder Information", "Coverage Description", "Vehicle Information",
        "Dealer/Seller Information", "form fields", "key-value pairs",
        "CSV DOCUMENT", "COLUMN INFORMATION", "STRUCTURED DATA RECORDS"
    ]
    enhanced_terms.extend(document_context_terms)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in enhanced_terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique_terms.append(term)
    
    return unique_terms, matched_categories, confidence_score

def perform_multi_strategy_search(user_question, enhanced_terms, matched_categories):
    """Comprehensive search strategy for maximum accuracy"""
    
    # Generate embeddings for the question
    question_embedding = get_embeddings(user_question)
    all_results = []
    
    # Safe fields that exist in your index
    safe_fields = ["content", "source_file", "file_type"]
    
    # Strategy 1: High-coverage vector search
    vector_query_comprehensive = VectorizedQuery(
        vector=question_embedding,
        k_nearest_neighbors=25,  # High coverage for accuracy
        fields="content_vector"
    )
    
    primary_search_terms = enhanced_terms[:20]  # More terms for better accuracy
    search_query = " OR ".join([f'"{term}"' for term in primary_search_terms])
    
    # Primary hybrid search
    try:
        primary_results = search_client.search(
            search_text=search_query,
            vector_queries=[vector_query_comprehensive],
            select=safe_fields,
            top=30  # More results for better coverage
        )
        all_results.extend(list(primary_results))
        logging.info(f"Primary search returned {len(list(primary_results))} results")
    except Exception as e:
        logging.error(f"Primary search failed: {e}")
    
    # Strategy 2: Exact field name search
    exact_field_terms = [term for term in enhanced_terms if 
                        len(term.split()) <= 4 and 
                        any(keyword in term.lower() for keyword in ['holder', 'agreement', 'vehicle', 'dealer', 'data', 'column'])]
    
    if exact_field_terms:
        try:
            exact_query = " OR ".join([f'"{term}"' for term in exact_field_terms[:15]])
            exact_results = search_client.search(
                search_text=exact_query,
                select=safe_fields,
                top=20
            )
            all_results.extend(list(exact_results))
            logging.info(f"Exact field search returned results")
        except Exception as e:
            logging.error(f"Exact field search failed: {e}")
    
    # Strategy 3: Fallback broad search
    if len(all_results) < 15:
        try:
            fallback_query = " OR ".join(user_question.split())
            fallback_results = search_client.search(
                search_text=fallback_query,
                vector_queries=[vector_query_comprehensive],
                select=safe_fields,
                top=20
            )
            all_results.extend(list(fallback_results))
            logging.info(f"Fallback search returned results")
        except Exception as e:
            logging.error(f"Fallback search failed: {e}")
    
    return all_results

def detect_primary_file_type(search_results):
    """Detect the primary file type from search results"""
    file_type_counts = {}
    for result in search_results:
        file_type = result.get('file_type', 'unknown')
        file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
    
    # Return the most common file type
    if file_type_counts:
        primary_type = max(file_type_counts, key=file_type_counts.get)
        logging.info(f"Primary file type detected: {primary_type}")
        return primary_type
    return 'unknown'

def intelligent_chunk_prioritization_for_forms(search_results, matched_categories):
    """Advanced chunk prioritization for form documents"""
    
    # Define priority weights
    priority_chunks = {
        'section_match': [],      # Weight: 10
        'form_fields': [],        # Weight: 9
        'exact_field': [],        # Weight: 8
        'structured_data': [],    # Weight: 7
        'relevant_content': [],   # Weight: 6
        'general_content': []     # Weight: 5
    }
    
    section_keywords = {
        'agreement_holder': ['agreement holder information', 'customer information'],
        'vehicle': ['vehicle information', 'auto information'],
        'dealer': ['dealer/seller information', 'seller information'],
        'coverage': ['coverage description', 'policy information']
    }
    
    for result in search_results:
        content = result['content'].lower()
        chunk_score = 0
        
        # Check for section matches (highest priority)
        for category in matched_categories:
            category_key = category.split('_')[0]  # Extract base category
            if category_key in section_keywords:
                for keyword in section_keywords[category_key]:
                    if keyword in content:
                        priority_chunks['section_match'].append((result, 10))
                        chunk_score = 10
                        break
                if chunk_score > 0:
                    break
        
        # Check for form fields and structured data
        if chunk_score == 0:
            if any(indicator in content for indicator in [
                '=== form fields', '=== key information', '•', 'field:',
                'agreement holder:', 'dealer/seller:', 'vehicle information:'
            ]):
                priority_chunks['form_fields'].append((result, 9))
                chunk_score = 9
        
        # Check for exact field matches
        if chunk_score == 0:
            if any(field in content for field in [
                'agreement holder', 'boris singh', 'home telephone', 
                'business telephone', 'vehicle identification'
            ]):
                priority_chunks['exact_field'].append((result, 8))
                chunk_score = 8
        
        # Check for structured data
        if chunk_score == 0:
            if any(indicator in content for indicator in [
                'row', 'col', 'table', '---', '|', 'address:', 'phone:'
            ]):
                priority_chunks['structured_data'].append((result, 7))
                chunk_score = 7
        
        # Default categorization
        if chunk_score == 0:
            if len(content.strip()) > 100:  # Meaningful content
                priority_chunks['relevant_content'].append((result, 6))
            else:
                priority_chunks['general_content'].append((result, 5))
    
    # Flatten prioritized results
    final_results = []
    for category in ['section_match', 'form_fields', 'exact_field', 'structured_data', 'relevant_content', 'general_content']:
        final_results.extend([item[0] for item in priority_chunks[category]])
    
    return final_results

def intelligent_chunk_prioritization_for_csv(search_results, user_question):
    """Specialized chunk prioritization for CSV data"""
    
    priority_chunks = {
        'data_records': [],       # Actual data rows
        'column_info': [],        # Column definitions
        'statistics': [],         # Summary statistics
        'key_value_pairs': [],    # Searchable pairs
        'general': []             # Other content
    }
    
    question_lower = user_question.lower()
    
    for result in search_results:
        content = result['content'].lower()
        
        # Prioritize based on CSV content type
        if 'record' in content and ('row' in content or ':' in content):
            priority_chunks['data_records'].append(result)
        elif 'column' in content and ('information' in content or 'summary' in content):
            priority_chunks['column_info'].append(result)
        elif any(stat in content for stat in ['average', 'min', 'max', 'total', 'count', 'summary']):
            priority_chunks['statistics'].append(result)
        elif 'contains' in content or 'key-value' in content:
            priority_chunks['key_value_pairs'].append(result)
        else:
            priority_chunks['general'].append(result)
    
    # Return prioritized results based on query type
    if any(word in question_lower for word in ['find', 'search', 'show', 'records']):
        # Prioritize data records for search queries
        return (priority_chunks['data_records'] + priority_chunks['key_value_pairs'] + 
                priority_chunks['column_info'] + priority_chunks['statistics'] + 
                priority_chunks['general'])
    elif any(word in question_lower for word in ['count', 'total', 'how many', 'statistics']):
        # Prioritize statistics for count queries
        return (priority_chunks['statistics'] + priority_chunks['column_info'] + 
                priority_chunks['data_records'] + priority_chunks['key_value_pairs'] + 
                priority_chunks['general'])
    else:
        # Default prioritization
        return (priority_chunks['column_info'] + priority_chunks['data_records'] + 
                priority_chunks['statistics'] + priority_chunks['key_value_pairs'] + 
                priority_chunks['general'])

def build_comprehensive_context_for_forms(prioritized_results, max_chunks=25):
    """Build rich context for form documents"""
    context_parts = []
    sources = set()
    metadata_info = []
    
    # Remove duplicates while preserving priority order
    seen_content_hashes = set()
    unique_results = []
    
    for result in prioritized_results:
        content_hash = hash(result['content'][:150])
        if content_hash not in seen_content_hashes:
            seen_content_hashes.add(content_hash)
            unique_results.append(result)
            if len(unique_results) >= max_chunks:
                break
    
    # Build comprehensive context
    context_parts.append("=== DOCUMENT ANALYSIS CONTEXT ===\n")
    
    for i, result in enumerate(unique_results):
        content = result['content']
        source_file = result.get('source_file', 'Unknown')
        file_type = result.get('file_type', 'document')
        chunk_seq = result.get('chunk_seq', i)
        
        # Track sources and metadata
        sources.add(f"{source_file} ({file_type})")
        metadata_info.append({
            'chunk': i+1,
            'source': source_file,
            'type': file_type,
            'sequence': chunk_seq
        })
        
        # Add chunk with rich metadata
        context_parts.append(f"--- CHUNK {i+1} | Source: {source_file} | Type: {file_type} | Sequence: {chunk_seq} ---")
        context_parts.append(content.strip())
        context_parts.append("")
    
    context_parts.append(f"\n=== ANALYSIS SUMMARY ===")
    context_parts.append(f"Total chunks analyzed: {len(unique_results)}")
    context_parts.append(f"Sources: {', '.join(sources)}")
    context_parts.append("")
    
    return "\n".join(context_parts), sources, metadata_info

def build_comprehensive_context_for_csv(prioritized_results, max_chunks=20):
    """Build specialized context for CSV data"""
    context_parts = []
    sources = set()
    
    # Remove duplicates
    seen_content_hashes = set()
    unique_results = []
    
    for result in prioritized_results:
        content_hash = hash(result['content'][:150])
        if content_hash not in seen_content_hashes:
            seen_content_hashes.add(content_hash)
            unique_results.append(result)
            if len(unique_results) >= max_chunks:
                break
    
    context_parts.append("=== CSV DATA ANALYSIS CONTEXT ===\n")
    
    for i, result in enumerate(unique_results):
        content = result['content']
        source_file = result.get('source_file', 'Unknown')
        sources.add(source_file)
        
        context_parts.append(f"--- Data Section {i+1} from {source_file} ---")
        context_parts.append(content.strip())
        context_parts.append("")
    
    context_parts.append(f"\n=== SUMMARY ===")
    context_parts.append(f"Total data sections analyzed: {len(unique_results)}")
    context_parts.append(f"CSV Sources: {', '.join(sources)}")
    context_parts.append("")
    
    return "\n".join(context_parts), sources

def create_form_system_prompt():
    """Comprehensive system prompt for form documents"""
    return """You are an expert AI assistant specialized in analyzing structured documents, particularly Vehicle Service Agreements, contracts, invoices, and forms. You have access to a comprehensive document context with multiple chunks of information.

## CORE CAPABILITIES:
- Extract specific field values from structured documents
- Understand document sections and their relationships
- Provide precise, factual answers based solely on provided context
- Handle complex queries about agreement holders, vehicles, dealers, coverage details

## DOCUMENT STRUCTURE EXPERTISE:

### Agreement Holder Information Section:
- Look for: "Agreement Holder", "Customer Information", "Subscriber Details"
- Key fields: Name, Address, Home Telephone, Business Telephone
- Example: "Agreement Holder: Boris Singh"

### Vehicle Information Section:
- Look for: "Vehicle Information", "Auto Details"
- Key fields: Year, Make, Model, VIN, Current Odometer, Purchase Date
- Example: "2016 Toyota Land Cruiser"

### Dealer/Seller Information Section:
- Look for: "Dealer/Seller Information", "Seller Details"
- Key fields: Dealer Name, Address, Phone, Dealer Number

### Coverage Description Section:
- Look for: "Coverage Description", "Agreement Details"
- Key fields: Agreement Number, Term, Purchase Date, Expiration Date, Deductible

## RESPONSE STRATEGY:

### For Field Extraction Queries:
1. **Identify the field type** from the question
2. **Locate relevant sections** in the provided chunks
3. **Extract exact values** from field-value pairs
4. **Provide complete information** with context

### Response Format Guidelines:
- **Direct Questions**: "What is [field]?" → "The [field] is [exact value]"
- **Complex Questions**: Provide comprehensive answers with all related information
- **Missing Information**: Clearly state if information is not found
- **Confidence**: Include source reference when helpful

### Quality Standards:
- Use ONLY information from provided context chunks
- Provide exact values, not paraphrases
- Include complete information (e.g., full address, not just street)
- Reference specific document sections when relevant
- Handle multi-part answers (e.g., name + address + phone)

## EXAMPLE INTERACTIONS:

**Query**: "What is agreement holder name?"
**Response**: "The Agreement Holder name is Boris Singh (as specified in the Agreement Holder Information section)."

**Query**: "What is the agreement holder's address?"
**Response**: "The Agreement Holder's address is 8818 Lew Wallace Rd, Frederick, Maryland 21704."

**Query**: "Tell me about the vehicle"
**Response**: "The vehicle is a 2016 Toyota Land Cruiser with VIN JTMCY7AJ8G4038164. The current odometer reading is 88,798 miles, and the vehicle purchase date was 01-01-2024."

## CRITICAL INSTRUCTIONS:
- Analyze ALL provided chunks systematically
- Prioritize information from form fields and structured sections
- If multiple chunks contain the same information, use the most complete version
- For questions about specific fields, provide exact field values
- If information spans multiple fields, combine them appropriately"""

def create_csv_system_prompt():
    """Specialized system prompt for CSV data analysis"""
    return """You are an expert AI assistant specialized in analyzing CSV/tabular data and answering questions based on structured datasets.

## CORE CAPABILITIES:
- Analyze CSV data records, columns, and relationships
- Extract specific values from data rows
- Provide statistical summaries and data insights
- Handle search queries across tabular data
- Understand column structures and data types

## CSV DATA UNDERSTANDING:

### Data Structure Recognition:
- **Column Information**: Field names, data types, structure
- **Data Records**: Individual rows with field values
- **Summary Statistics**: Aggregated data insights
- **Key-Value Pairs**: Searchable data relationships

### Query Types and Responses:

#### Search Queries ("Find records where...", "Show me data with..."):
- Look through data records for matching values
- Provide specific row data that matches criteria
- Include column names and corresponding values
- Reference record numbers when available

#### Count/Statistics Queries ("How many...", "What's the average..."):
- Use summary statistics sections
- Provide numerical insights from aggregated data
- Include relevant statistical measures
- Reference data sources and sample sizes

#### Column/Structure Queries ("What columns...", "What data is available..."):
- Describe available fields and data types
- Explain data structure and organization
- List unique values or data ranges
- Provide column summaries

#### Data Analysis Queries ("Analyze...", "Compare...", "Show trends..."):
- Combine multiple data points for insights
- Provide comprehensive data analysis
- Use statistical summaries effectively
- Draw conclusions from available data

## RESPONSE GUIDELINES:

### Accuracy Standards:
- Use ONLY information from provided CSV data sections
- Quote exact values from data records
- Provide specific record references when possible
- Include column names for clarity

### Format Guidelines:
- Use bullet points for multiple results
- Include data source references
- Provide clear, structured responses
- Use tables or lists for complex data

### Missing Data Handling:
- Clearly state if requested information is not available
- Suggest alternative queries if data exists
- Explain data limitations when relevant

## EXAMPLE INTERACTIONS:

**Query**: "Find all records where department is Sales"
**Response**: "Based on the CSV data, here are the records where department = Sales:
• Record 3: Name = John Smith, Department = Sales, Salary = $65,000
• Record 7: Name = Sarah Johnson, Department = Sales, Salary = $58,000"

**Query**: "What's the average salary?"
**Response**: "According to the data summary, the average salary is $62,450 based on 50 employee records."

**Query**: "What columns are available?"
**Response**: "The CSV contains the following columns:
• Name (employee names)
• Department (work departments)
• Salary (annual compensation)
• Start Date (employment start dates)"

## CRITICAL INSTRUCTIONS:
- Analyze all provided data sections systematically
- Prioritize exact data matches for search queries
- Use statistical sections for numerical queries
- Provide complete, accurate information only
- Reference specific data sources when helpful"""

# === Document/Query Type Understanding (Enhanced) ===
def create_query_understanding_mapping():
    """Provides context for LLM-based query type detection and keyword expansion."""
    return {
        'form_document_queries': {
            'keywords': ['agreement holder', 'vehicle information', 'vin', 'deductible', 'policy number', 'dealer name', 'form field'],
            'description': 'Questions about specific fields, sections, or terms within structured forms like Vehicle Service Agreements, contracts, or detailed PDF forms.',
            'target_strategy': 'document_search_specialized_form'
        },
        'csv_data_queries': {
            'keywords': ['csv data', 'table data', 'column statistics', 'average value', 'total records', 'filter rows'],
            'description': 'Questions about data within CSV files, including specific record lookups, aggregations, or column information.',
            'target_strategy': 'document_search_specialized_csv'
        },
        'cosmosdb_queries': {
            'keywords': ['customer data', 'product catalog', 'sales orders', 'adventureworks items', 'database records', 'count products'],
            'description': 'Questions about structured data stored in the AdventureWorks NoSQL database (Cosmos DB), like customer details, product listings, or sales information.',
            'target_strategy': 'cosmosdb_query'
        },
        'general_document_queries': {
            'keywords': ['summarize document', 'main points of file', 'content of pdf', 'details in json'],
            'description': 'General questions about the content of uploaded PDF or JSON documents that are not specific forms or structured CSVs.',
            'target_strategy': 'document_search_generic'
        }
    }

def discover_database_schema():
    """Dynamically discover the actual database schema"""
    try:
        # Sample a few documents to understand structure
        sample_query = "SELECT TOP 5 * FROM c"
        execution_response = call_query_cosmos_db_function(sample_query)
        
        if execution_response and execution_response.get("success"):
            sample_data = execution_response.get("data", [])
            if sample_data:
                # Analyze the first document to understand schema
                first_doc = sample_data[0]
                fields = list(first_doc.keys())
                
                schema_summary = f"""
Database Schema (Auto-discovered from {len(sample_data)} sample documents):

Document Structure:
"""
                for field in fields:
                    sample_value = first_doc.get(field)
                    value_type = type(sample_value).__name__
                    schema_summary += f"- {field}: {value_type} (example: {sample_value})\n"
                
                schema_summary += f"""
Example Queries:
- Find by category: SELECT * FROM c WHERE c.category = 'your_category'
- Find by id: SELECT * FROM c WHERE c.id = 'your_id'
- Count all: SELECT VALUE COUNT(1) FROM c
- Group by category: SELECT c.category, COUNT(1) as count FROM c GROUP BY c.category

Query Notes:
- Use 'c' as container alias
- String values need single quotes
- Use exact field names as shown above
"""
                return schema_summary.strip()
        
        # Fallback to manual schema if auto-discovery fails
        return get_manual_schema_for_your_db()
        
    except Exception as e:
        logging.error(f"Error discovering schema: {e}")
        return get_manual_schema_for_your_db()

def get_manual_schema_for_your_db():
    """Manual schema definition for your specific database"""
    return f"""
Database Schema (Manual Configuration):
Container: {COSMOS_CONTAINER_NAME_FOR_PROMPT}
Database: {COSMOS_DATABASE_NAME_FOR_PROMPT}

Document Structure:
- id: string (partition key) - values: "1", "2", "3", etc.
- category: string - values: "a", "b", "c", etc.

Example Document:
{{"id": "1", "category": "a"}}

Query Examples:
- Find items with category 'c': SELECT * FROM c WHERE c.category = 'c'
- Find specific id: SELECT * FROM c WHERE c.id = '2'
- List all categories: SELECT DISTINCT c.category FROM c
- Count by category: SELECT c.category, COUNT(1) FROM c GROUP BY c.category

Important:
- Use 'c' as the container alias
- String values must be in single quotes
- Field names are case-sensitive
"""

CACHED_COSMOS_SCHEMA_SUMMARY = None
def get_cosmos_schema_summary_for_prompt(force_refresh=False):
    global CACHED_COSMOS_SCHEMA_SUMMARY
    if CACHED_COSMOS_SCHEMA_SUMMARY and not force_refresh:
        return CACHED_COSMOS_SCHEMA_SUMMARY
    
    summary = f"""
The NoSQL database (Azure Cosmos DB: Container '{COSMOS_CONTAINER_NAME_FOR_PROMPT}' in Database '{COSMOS_DATABASE_NAME_FOR_PROMPT}') contains AdventureWorksLT data.
Documents generally have an 'id' (partition key) and a 'documentType' field (e.g., "Product", "Customer", "SalesOrderHeader").

Key Document Structures:
1. Product:
   - Fields: ProductID, Name, ProductNumber, Color, StandardCost, ListPrice, Size, Weight, ProductCategoryID, ProductModelID, SellStartDate, Description (if added).
   - Example Query: SELECT * FROM c WHERE c.documentType = "Product" AND c.Color = 'Black'

2. Customer:
   - Fields: CustomerID, FirstName, LastName, EmailAddress, Phone, Addresses (array of nested address objects: AddressType, AddressLine1, City, StateProvince, PostalCode).
   - Example Query: SELECT * FROM c WHERE c.documentType = "Customer" AND c.LastName = 'Yang'

3. SalesOrderHeader:
   - Fields: SalesOrderID, OrderDate, CustomerID, TotalDue, OrderDetails (array of nested line items: ProductID, OrderQty, UnitPrice, LineTotal).
   - Example Query: SELECT * FROM c WHERE c.documentType = "SalesOrderHeader" AND c.SalesOrderID = 71774

Querying Notes:
- Use 'c' as the alias for the container (e.g., `SELECT c.Name FROM c`).
- String values in filters must be in single quotes.
- For counts: `SELECT VALUE COUNT(1) FROM c WHERE ...`
- To query nested array elements, you might need JOIN (e.g., `SELECT VALUE od FROM c JOIN od IN c.OrderDetails WHERE c.SalesOrderID = 123`). Prefer simpler queries if possible for NL2Query.
"""
    CACHED_COSMOS_SCHEMA_SUMMARY = summary.strip()
    logging.info("Loaded Cosmos DB schema summary for LLM prompt.")
    return CACHED_COSMOS_SCHEMA_SUMMARY

def llm_determine_query_strategy(user_question: str) -> dict:
    if not openai_client:
        return {"strategy": "error", "reasoning": "OpenAI client not initialized."}
    
    query_type_map_description = "Available data sources and query types:\n"
    for cat, details in create_query_understanding_mapping().items():
        query_type_map_description += f"- Type '{details['target_strategy']}': {details['description']}. Keywords: {', '.join(details['keywords'][:3])}...\n"
    
    cosmos_schema_summary = discover_database_schema()

    detection_prompt = f"""
Analyze the user's question and determine the most appropriate strategy to answer it.
You have access to general PDF/JSON documents, specific PDF forms (like Vehicle Service Agreements), CSV tabular data, and a structured NoSQL database (Cosmos DB containing AdventureWorks data).

User Question: "{user_question}"

Consider the following data source types and query strategies:
{query_type_map_description}

Cosmos DB (AdventureWorks Data) Schema Summary (if relevant for 'cosmosdb_query'):
---
{cosmos_schema_summary}
---

Based on the user's question, choose the BEST single strategy from:
- "document_search_specialized_form" (for VSA-like forms)
- "document_search_specialized_csv" (for CSV data)
- "cosmosdb_query" (for AdventureWorks data in NoSQL DB)
- "document_search_generic" (for other general PDF/JSON documents)
- "hybrid_all" (ONLY if the question explicitly asks to combine information from distinctly different sources like a PDF AND the database, or if it's highly ambiguous).

Respond with ONLY a JSON object with "strategy", "confidence" (0.0-1.0), and "reasoning".
Example: {{"strategy": "cosmosdb_query", "confidence": 0.9, "reasoning": "Question asks for specific product details likely in the database."}}
"""
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_DEPLOYMENT_FOR_INTENT,
            messages=[
                {"role": "system", "content": "You are an AI assistant that classifies user questions into data retrieval strategies."},
                {"role": "user", "content": detection_prompt}
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        decision_str = response.choices[0].message.content
        logging.info(f"LLM Query Strategy Detection response: {decision_str}")
        decision = json.loads(decision_str)
        if decision.get("strategy") in ["document_search_specialized_form", "document_search_specialized_csv", "cosmosdb_query", "document_search_generic", "hybrid_all"]:
            return decision
        else:
            logging.warning(f"LLM strategy detection returned unexpected strategy: {decision.get('strategy')}")
            return {"strategy": "document_search_generic", "confidence": 0.4, "reasoning": "Fallback due to unexpected LLM strategy."}
    except Exception as e:
        logging.error(f"Error in LLM query strategy detection: {str(e)}. Defaulting.")
        return {"strategy": "document_search_generic", "confidence": 0.3, "reasoning": f"Error in LLM detection: {str(e)}."}

def convert_nl_to_cosmos_query(user_question, db_schema_for_prompt):
    if not openai_client:
        return None
    
    system_prompt_nl2cosmos = f"""You are an expert Azure Cosmos DB query programmer for the API for NoSQL (SQL-like syntax).
Your task is to convert the user's natural language question into a single, valid Cosmos DB SELECT query.
Strictly adhere to all rules provided (SELECT only, no markdown, INVALID_QUERY for unanswerable, use 'c' alias, etc.).

Database Schema Summary:
---
{db_schema_for_prompt}
---
User Question: {user_question}
Cosmos DB Query:"""

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_DEPLOYMENT_FOR_NL2COSMOS,
            messages=[
                {"role": "system", "content": "You are an expert Azure Cosmos DB query programmer."},
                {"role": "user", "content": system_prompt_nl2cosmos}
            ],
            temperature=0.0,
            max_tokens=350,
            top_p=0.1
        )
        cosmos_query = response.choices[0].message.content.strip()
        logging.info(f"LLM generated Cosmos DB query: {cosmos_query}")
        if "INVALID_QUERY" in cosmos_query or not cosmos_query.upper().startswith("SELECT"):
            return None
        return cosmos_query.replace("```sql", "").replace("```json", "").replace("```", "").strip()
    except Exception as e:
        logging.error(f"Error converting NL to Cosmos DB query: {str(e)}")
        return None

def call_query_cosmos_db_function(query):
    if not QUERY_COSMOSDB_FUNCTION_URL:
        logging.error("QUERY_COSMOSDB_FUNCTION_URL not configured.")
        return {"success": False, "error": "Cosmos DB execution endpoint not configured."}
    
    headers = {'Content-Type': 'application/json'}
    if QUERY_COSMOSDB_FUNCTION_KEY:
        headers['x-functions-key'] = QUERY_COSMOSDB_FUNCTION_KEY
    
    payload = {"query": query}
    try:
        logging.info(f"Calling QueryCosmosDB function with query: {query}")
        response = requests.post(QUERY_COSMOSDB_FUNCTION_URL, json=payload, headers=headers, timeout=45)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error calling QueryCosmosDB: {http_err}. Response: {http_err.response.text if http_err.response else 'No response text'}")
        try:
            return http_err.response.json()
        except:
            return {"success": False, "error": f"CosmosDB function HTTP error: {http_err.response.status_code if http_err.response else 'Unknown'}"}
    except requests.exceptions.RequestException as e:
        logging.error(f"RequestException calling QueryCosmosDB: {str(e)}")
        return {"success": False, "error": f"Connection error to CosmosDB function: {str(e)}"}

# === Main Function ===
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('AskQuestion function (PDF/CSV/JSON RAG + CosmosDB NLQ) processing request.')
    
    # Initial validation
    if not search_client or not openai_client:
        return func.HttpResponse(
            json.dumps({"error": "Core services not initialized."}),
            status_code=503,
            mimetype="application/json"
        )

    try:
        req_body = req.get_json()
        user_question = req_body.get('question')
        query_mode_hint = req_body.get('query_mode', 'auto_detect')
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON"}),
            status_code=400,
            mimetype="application/json"
        )

    if not user_question:
        return func.HttpResponse(
            json.dumps({"error": "Question missing"}),
            status_code=400,
            mimetype="application/json"
        )

    try:
        document_rag_context_str = ""
        cosmos_query_results_str = ""
        sources = set()
        final_answer = "I was unable to find an answer to your question using the available information."
        
        # MODIFICATION STARTS HERE
        if query_mode_hint == 'document_search_generic':
            effective_strategy = 'document_search_generic'
            response_metadata = {
                "strategy_used": effective_strategy,
                "llm_detection_reasoning": "User selected 'Documents' query target."
            }
        elif query_mode_hint == 'cosmosdb_query':
            effective_strategy = 'cosmosdb_query'
            response_metadata = {
                "strategy_used": effective_strategy,
                "llm_detection_reasoning": "User selected 'NoSQL DB' query target."
            }
        else:  # This will handle 'hybrid_all' and 'auto_detect'
            cosmos_schema_for_detection = discover_database_schema()
            strategy_decision = llm_determine_query_strategy(user_question)
            effective_strategy = strategy_decision.get("strategy", "document_search_generic")
            response_metadata = {
                "strategy_used": effective_strategy,
                "llm_detection_reasoning": strategy_decision.get("reasoning")
            }
        
        logging.info(f"Effective strategy: {effective_strategy} (Hint from frontend: {query_mode_hint})")
        # MODIFICATION ENDS HERE

        # Document Search Path
        if effective_strategy.startswith("document_search"):
            try:
                enhanced_terms, matched_categories, confidence_score = enhance_query_with_comprehensive_context(user_question)
                search_results = perform_multi_strategy_search(user_question, enhanced_terms, matched_categories)
                
                if search_results:
                    primary_file_type = detect_primary_file_type(search_results)
                    response_metadata["detected_document_source_type"] = primary_file_type

                    if primary_file_type == 'csv' and effective_strategy == "document_search_specialized_csv":
                        chunks = intelligent_chunk_prioritization_for_csv(search_results, user_question)
                        document_rag_context_str, doc_sources = build_comprehensive_context_for_csv(chunks)
                    elif primary_file_type in ['pdf', 'json'] and effective_strategy == "document_search_specialized_form":
                        chunks = intelligent_chunk_prioritization_for_forms(search_results, matched_categories)
                        document_rag_context_str, doc_sources, _ = build_comprehensive_context_for_forms(chunks)
                    else:
                        chunks = intelligent_chunk_prioritization_for_forms(search_results, matched_categories)
                        document_rag_context_str, doc_sources, _ = build_comprehensive_context_for_forms(chunks, max_chunks=15)
                    
                    sources.update(doc_sources)
                else:
                    logging.info("No relevant document search results found.")
                    document_rag_context_str = ""
            except Exception as e_doc:
                logging.error(f"Error in document search: {str(e_doc)}", exc_info=True)
                document_rag_context_str = "An error occurred while searching documents."

        # Cosmos DB Query Path
        if effective_strategy == "cosmosdb_query" or effective_strategy == "hybrid_all":
            if not QUERY_COSMOSDB_FUNCTION_URL or not COSMOS_ENDPOINT_FOR_PROMPT:
                cosmos_query_results_str = "Cosmos DB querying is not configured for this system."
            else:
                try:
                    generated_query = convert_nl_to_cosmos_query(user_question, cosmos_schema_for_detection)
                    if generated_query:
                        execution_response = call_query_cosmos_db_function(generated_query)
                        if execution_response and execution_response.get("success"):
                            cosmos_data = execution_response.get("data", [])
                            if cosmos_data:
                                cosmos_query_results_str = f"Data from AdventureWorks NoSQL Database (query: {generated_query}):\n"
                                for i, item in enumerate(cosmos_data[:5]):
                                    cosmos_query_results_str += f"Item {i+1}: {json.dumps(item, indent=2)}\n"
                                if len(cosmos_data) > 5:
                                    cosmos_query_results_str += f"...and {len(cosmos_data)-5} more items.\n"
                                sources.add("AdventureWorks Data (Cosmos DB)")
                            else:
                                cosmos_query_results_str = f"The query returned no data: {generated_query}"
                        else:
                            error_msg = execution_response.get('error', 'Unknown error') if execution_response else 'No response'
                            cosmos_query_results_str = f"Query execution error: {error_msg}"
                    else:
                        cosmos_query_results_str = "Could not generate a valid database query for your question."
                except Exception as e_cosmos:
                    logging.error(f"Error in Cosmos DB query: {str(e_cosmos)}", exc_info=True)
                    cosmos_query_results_str = f"Error processing database query: {str(e_cosmos)}"

        # Generate Final Answer
        final_context = ""
        if document_rag_context_str and not document_rag_context_str.startswith("An error"):
            final_context += f"DOCUMENT CONTEXT:\n{document_rag_context_str}\n\n"
        if cosmos_query_results_str and not cosmos_query_results_str.startswith(("Error", "Could not", "Failed")):
            final_context += f"DATABASE CONTEXT:\n{cosmos_query_results_str}\n\n"

        if not final_context:
            if effective_strategy == "cosmosdb_query" and cosmos_query_results_str:
                final_answer = cosmos_query_results_str
            elif effective_strategy.startswith("document_search") and document_rag_context_str:
                final_answer = document_rag_context_str
            else:
                final_answer = "I could not find relevant information to answer your question."
        else:
            try:
                # Prepare the appropriate prompt based on the context type
                if effective_strategy == "document_search_specialized_form":
                    system_prompt = create_form_system_prompt()
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"DOCUMENT CONTEXT:\n{final_context}\n\nQUESTION: {user_question}"}
                    ]
                elif effective_strategy == "document_search_specialized_csv":
                    system_prompt = create_csv_system_prompt()
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"CSV DATA:\n{final_context}\n\nQUESTION: {user_question}"}
                    ]
                else:
                    messages = [{
                        "role": "system",
                        "content": """You are an AI assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information to answer the question, say so.
Do not use external knowledge or make assumptions."""
                    }, {
                        "role": "user",
                        "content": f"CONTEXT:\n{final_context}\n\nQUESTION: {user_question}"
                    }]

                response = openai_client.chat.completions.create(
                    model=OPENAI_CHAT_DEPLOYMENT,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2000,
                    top_p=0.1
                )
                final_answer = response.choices[0].message.content.strip()
            except Exception as e_answer:
                logging.error(f"Error generating final answer: {str(e_answer)}", exc_info=True)
                final_answer = "Sorry, I encountered an error while formulating the answer."
                response_metadata["error"] = str(e_answer)

        return func.HttpResponse(
            json.dumps({
                "answer": final_answer,
                "sources": sorted(list(sources)),
                "metadata": response_metadata
            }, indent=2),
            mimetype="application/json"
        )

    except Exception as e:
        logging.critical(f"Critical error in AskQuestion: {str(e)}", exc_info=True)
        return func.HttpResponse(
            json.dumps({
                "error": "An unexpected error occurred.",
                "answer": "I'm sorry, I encountered an error while processing your question.",
                "sources": [],
                "metadata": {"strategy_used": "error"}
            }),
            status_code=500,
            mimetype="application/json"
        )
