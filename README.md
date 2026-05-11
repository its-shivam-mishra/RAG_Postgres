# Docura RAG - Document Q&A System

A Retrieval-Augmented Generation (RAG) application built with FastAPI, Azure OpenAI, PostgreSQL with pgvector, and Tavily Search for intelligent document querying.

## Features

- **Document Upload**: Upload PDF and TXT files to build a knowledge base
- **Intelligent Search**: Powered by Azure OpenAI embeddings and pgvector for semantic search
- **Web Search Integration**: Enhances responses with Tavily Search for up-to-date information
- **User Authentication**: Secure login via Azure SSO (Microsoft Entra ID)
- **Modern Web Interface**: Clean, responsive UI for document management and Q&A
- **API-First Design**: RESTful APIs for easy integration
- **CI/CD Ready**: Azure Pipelines configuration for automated deployment

## Architecture

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL with pgvector extension for vector storage
- **AI Services**: Azure OpenAI for embeddings and chat completion
- **Search**: Tavily API for web search augmentation
- **Frontend**: Vanilla JavaScript with modern CSS
- **Authentication**: Azure SSO integration

## High-Level Data Flow

The application follows a typical RAG (Retrieval-Augmented Generation) pattern with the following data flow:

```mermaid
graph TD
    A[User Uploads Document] --> B[Document Processing]
    B --> C[Text Extraction & Chunking]
    C --> D[Generate Embeddings via Azure OpenAI]
    D --> E[Store Vectors in PostgreSQL pgvector]
    
    F[User Asks Question] --> G[Authenticate via Azure SSO]
    G --> H[Generate Question Embedding]
    H --> I[Vector Similarity Search]
    I --> J[Retrieve Relevant Document Chunks]
    
    J --> K[Optional: Web Search via Tavily]
    K --> L[Combine Context]
    
    L --> M[Generate Answer via Azure OpenAI]
    M --> N[Return Response to User]
    
    E --> I
```

### Data Flow Steps:

1. **Document Ingestion**:
   - User uploads PDF/TXT files via web interface
   - `app/routes/documents.py::upload_file()` handles the upload endpoint
   - `app/document.py::ingest_document()` processes the file: text extraction using PyMuPDF, chunking with RecursiveCharacterTextSplitter
   - Each chunk is converted to vector embeddings using Azure OpenAI's embedding model (via `app/models.py::embeddings`)
   - Embeddings are stored in PostgreSQL with pgvector using `langchain_postgres.vectorstores.PGVector`

2. **Query Processing**:
   - User submits a question through the chat interface
   - `app/auth.py::get_current_user()` dependency verifies authentication via Azure SSO
   - `app/routes/query.py::query_endpoint()` receives the query
   - `app/query.py::answer_question()` orchestrates the RAG pipeline
   - Question is converted to embedding and `app/retriever.py::retrieve_and_rerank()` performs vector similarity search with FlashRank reranking

3. **Answer Generation**:
   - Retrieved chunks form the context for the AI model
   - Optional web search via `app/search.py::tavily_search()` provides additional current information using Tavily API
   - `app/query.py::answer_question()` uses `langchain_classic.chains.combine_documents.create_stuff_documents_chain` with Azure OpenAI chat completion
   - Response is streamed back to the user interface

4. **Security & Access**:
   - All interactions require valid Azure SSO authentication via `app/auth.py::login()` and `app/auth.py::auth_callback()`
   - Session management ensures user context isolation using Starlette sessions
   - API endpoints are protected with proper authorization dependencies

## Prerequisites

- Python 3.13+
- PostgreSQL with pgvector extension
- Azure OpenAI resource
- Azure App Registration for SSO
- Tavily API key (optional, for web search)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/its-shivam-mishra/RAG_Postgres.git
   cd RAG_Postgres
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up PostgreSQL**:
   - Install PostgreSQL and enable pgvector extension
   - Create a database for the application

5. **Configure environment variables**:
   Create a `.env` file in the root directory:
   ```env
   # Database
   DATABASE_URL=postgresql://username:password@localhost:5432/vectordb

   # Azure OpenAI
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_KEY=your-azure-openai-key
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

   # Azure SSO
   AZURE_CLIENT_ID=your-client-id
   AZURE_CLIENT_SECRET=your-client-secret
   AZURE_TENANT_ID=your-tenant-id
   SESSION_SECRET_KEY=your-super-secret-key

   # Tavily Search (optional)
   TAVILY_API_KEY=your-tavily-api-key
   ```

## Running the Application

1. **Initialize the database**:
   The database schema is automatically created on startup.

2. **Run the development server**:
   ```bash
   python -m app.main
   ```
   The application will be available at `http://127.0.0.1:8000`

3. **Access the web interface**:
   - Open your browser and navigate to `http://127.0.0.1:8000`
   - Sign in with your Microsoft account
   - Upload documents and start asking questions

## API Endpoints

### Authentication
- `GET /api/auth/login` - Initiate Azure SSO login
- `GET /api/auth/callback` - SSO callback handler

### Documents
- `POST /api/documents/upload` - Upload a document
- `GET /api/documents/list` - List uploaded documents
- `DELETE /api/documents/{id}` - Delete a document

### Query
- `POST /api/query/ask` - Ask a question about the documents

## Deployment

The application includes Azure Pipelines configuration for automated deployment to Azure Web Apps.

### Azure Deployment

1. **Set up Azure resources**:
   - Azure Web App
   - Azure Database for PostgreSQL
   - Azure OpenAI resource
   - Azure App Registration

2. **Configure pipeline variables**:
   - Update `azureServiceConnectionId` in `azure-pipelines.yml`
   - Set environment variables in Azure Web App configuration

3. **Deploy**:
   - Push to the `main` branch to trigger automatic deployment

## Development

### Project Structure
```
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── database.py          # Database initialization
│   ├── models.py            # SQLAlchemy models
│   ├── auth.py              # Authentication logic
│   ├── document.py          # Document processing
│   ├── query.py             # Query processing
│   ├── rag_engine.py        # RAG logic
│   ├── retriever.py         # Vector retrieval
│   ├── search.py            # Search integration
│   └── routes/
│       ├── __init__.py
│       ├── documents.py     # Document API routes
│       └── query.py         # Query API routes
├── static/
│   ├── index.html           # Main web interface
│   ├── style.css            # CSS styles
│   └── script.js            # Frontend JavaScript
├── temp_uploads/            # Temporary file storage
├── requirements.txt         # Python dependencies
├── azure-pipelines.yml      # CI/CD pipeline
└── test_*.py               # Test files
```

### Running Tests
```bash
python -m pytest test_*.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open an issue on GitHub.