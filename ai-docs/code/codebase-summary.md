# Verba Codebase Summary

## Project Overview

**Verba (The Golden RAGtriever)** is an open-source Retrieval-Augmented Generation (RAG) application that provides a user-friendly interface for querying documents using various LLM providers. It combines state-of-the-art RAG techniques with Weaviate's vector database to enable semantic search and AI-powered question answering over custom datasets.

**Version:** 2.1.3
**Python Requirements:** >=3.10.0, <3.13.0
**Repository:** https://github.com/weaviate/Verba

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                    │
│  - React/TypeScript UI                                  │
│  - TailwindCSS + DaisyUI styling                        │
│  - WebSocket for real-time updates                      │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP/WebSocket
┌────────────────▼────────────────────────────────────────┐
│              FastAPI Backend Server                      │
│  - RESTful API endpoints                                │
│  - WebSocket connections                                │
│  - Client connection management                         │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│                  VerbaManager                           │
│  - Central orchestration layer                          │
│  - Component managers (Reader, Chunker, etc.)           │
│  - RAG pipeline coordination                            │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Weaviate Vector Database                    │
│  - Document storage                                     │
│  - Vector embeddings (multiple collections)             │
│  - Hybrid search (semantic + keyword)                   │
└─────────────────────────────────────────────────────────┘
```

---

## Backend Architecture (Python)

### Core Directory Structure

```
goldenverba/
├── server/              # FastAPI server and API endpoints
├── components/          # RAG pipeline components
│   ├── reader/         # Document ingestion (PDF, HTML, Git, etc.)
│   ├── chunking/       # Text chunking strategies
│   ├── embedding/      # Embedding model integrations
│   ├── retriever/      # Search/retrieval logic
│   ├── generation/     # LLM integrations
│   ├── document.py     # Document data model
│   ├── chunk.py        # Chunk data model
│   ├── interfaces.py   # Base component interfaces
│   └── managers.py     # Component managers
├── verba_manager.py    # Main orchestration class
└── tests/              # Unit tests
```

### Key Components

#### 1. **VerbaManager** (`verba_manager.py`)
The central orchestrator that manages all RAG pipeline components:

- **Component Managers:**
  - `ReaderManager` - Handles document loading
  - `ChunkerManager` - Manages text chunking
  - `EmbeddingManager` - Coordinates embeddings
  - `RetrieverManager` - Handles search/retrieval
  - `GeneratorManager` - Manages LLM generation
  - `WeaviateManager` - Database operations

- **Key Responsibilities:**
  - Configuration management (RAG, theme, user configs)
  - Document import pipeline orchestration
  - Async task coordination
  - Environment/library verification

#### 2. **Component Interface System** (`interfaces.py`)
Abstract base classes for pluggable components:

- **VerbaComponent**: Base class with metadata, config, and availability checking
- **Reader**: Document ingestion interface
- **Chunker**: Text splitting interface
- **Embedding**: Vector embedding interface
- **Retriever**: Search/retrieval interface
- **Generator**: LLM generation interface

Each component can be swapped out, allowing flexible RAG pipeline configuration.

#### 3. **WeaviateManager** (`managers.py`)
Handles all Weaviate database operations:

- **Collections:**
  - `VERBA_DOCUMENTS` - Document metadata
  - `VERBA_Embedding_*` - Chunk embeddings (per model)
  - `VERBA_CONFIGURATION` - System configs
  - `VERBA_SUGGESTIONS` - Query autocomplete

- **Key Operations:**
  - Connection management (Local, Docker, Cloud, Custom)
  - CRUD operations for documents/chunks
  - Hybrid search (semantic + BM25)
  - PCA generation for 3D vector visualization
  - Label filtering and suggestion tracking

#### 4. **Document Data Model** (`document.py`, `chunk.py`)

**Document:**
- Contains full document content and metadata
- Uses spaCy for language detection and sentence tokenization
- Supports batched processing for large documents
- Stores configuration metadata (reader, chunker, embedder)

**Chunk:**
- Represents a text segment with overlap support
- Contains vector embeddings and PCA coordinates
- Links back to parent document via UUID
- Tracks position indices for context retrieval

#### 5. **FastAPI Server** (`server/api.py`)
Main API server with:

- **Security:** Same-origin policy middleware
- **WebSocket:** Real-time communication for ingestion/generation
- **Lifecycle Management:** ClientManager for connection pooling
- **Static File Serving:** Serves Next.js frontend build

---

## RAG Pipeline Components

### Readers (Document Ingestion)
- **BasicReader**: Plain text, PDF, DOCX, CSV, XLSX
- **HTMLReader**: HTML files
- **GitReader**: GitHub/GitLab repositories
- **UnstructuredReader**: Uses Unstructured.io API for complex docs
- **AssemblyAIReader**: Audio transcription
- **FirecrawlReader**: Web scraping/crawling
- **UpstageDocumentParseReader**: Upstage Document AI

### Chunkers (Text Splitting)
- **TokenChunker**: Fixed token-size chunks (spaCy)
- **SentenceChunker**: Sentence-based chunking (spaCy)
- **RecursiveChunker**: Recursive character splitting
- **SemanticChunker**: Semantic similarity-based grouping
- **HTMLChunker**: Preserves HTML structure
- **MarkdownChunker**: Preserves Markdown structure
- **CodeChunker**: Language-aware code chunking
- **JSONChunker**: JSON structure-aware chunking

### Embedders (Vector Models)
- **OllamaEmbedder**: Local Ollama models
- **SentenceTransformersEmbedder**: HuggingFace models
- **WeaviateEmbedder**: Weaviate-hosted models
- **OpenAIEmbedder**: OpenAI embeddings (supports custom endpoints)
- **CohereEmbedder**: Cohere embeddings
- **VoyageAIEmbedder**: VoyageAI embeddings
- **UpstageEmbedder**: Upstage embeddings

### Retrievers (Search)
- **WindowRetriever**: Hybrid search with configurable window context

### Generators (LLMs)
- **OllamaGenerator**: Local Ollama models
- **OpenAIGenerator**: OpenAI GPT models
- **AnthropicGenerator**: Claude models
- **CohereGenerator**: Cohere Command models
- **GroqGenerator**: Groq LPU inference
- **NovitaGenerator**: Novita AI models
- **UpstageGenerator**: Upstage Solar models

---

## Frontend Architecture (Next.js/React)

### Directory Structure

```
frontend/
├── app/
│   ├── components/
│   │   ├── Chat/              # Chat interface
│   │   ├── Document/          # Document explorer
│   │   ├── Ingestion/         # File upload/config
│   │   ├── Login/             # Deployment setup
│   │   ├── Navigation/        # Navbar, status
│   │   └── Settings/          # Settings panel
│   ├── api.ts                 # API client functions
│   ├── types.ts               # TypeScript interfaces
│   ├── util.ts                # Utility functions
│   ├── page.tsx               # Main app component
│   └── layout.tsx             # App layout
├── tailwind.config.ts         # TailwindCSS config
└── package.json
```

### Key Features

- **Chat Interface:** RAG query UI with streaming responses
- **Document Explorer:** Browse/search documents, view chunks, 3D vector visualization
- **Ingestion View:** Upload files, configure RAG pipeline per file
- **Settings:** Configure API keys, RAG components, themes
- **Real-time Updates:** WebSocket for ingestion progress and generation streaming
- **Theme System:** Multiple customizable UI themes

---

## Data Flow

### Document Ingestion Pipeline

```
1. File Upload → 2. Reader → 3. Chunker → 4. Embedder → 5. Weaviate Storage

User uploads file
↓
Reader converts to Document object(s)
↓
Chunker splits into Chunk objects with overlap
↓
Embedder generates vectors for each chunk
↓
PCA computed for 3D visualization
↓
Document + Chunks stored in Weaviate
```

### Query Pipeline (RAG)

```
1. User Query → 2. Embed Query → 3. Hybrid Search → 4. Context Assembly → 5. LLM Generation

User asks question
↓
Query embedded to vector
↓
Weaviate hybrid search (semantic + BM25)
↓
Top chunks retrieved with context windows
↓
Context + query sent to Generator
↓
Streaming response to frontend
```

---

## Key Technologies

### Backend
- **FastAPI**: Web framework
- **Weaviate**: Vector database (v4.9.6)
- **spaCy**: NLP and sentence tokenization (v3.7.5)
- **scikit-learn**: PCA for vector visualization (v1.5.1)
- **LangChain**: Text splitting utilities (v0.2.2)
- **tiktoken**: Token counting (v0.6.0)
- **asyncio**: Async pipeline execution

### Frontend
- **Next.js**: React framework
- **TypeScript**: Type safety
- **TailwindCSS**: Styling
- **DaisyUI**: Component library

### Deployment
- **Docker**: Containerized deployment
- **Docker Compose**: Multi-container orchestration (Verba + Weaviate)

---

## Configuration System

### RAG Configuration
Stored in Weaviate (`VERBA_CONFIGURATION` collection):

```python
{
    "Reader": {"selected": "...", "components": {...}},
    "Chunker": {"selected": "...", "components": {...}},
    "Embedder": {"selected": "...", "components": {...}},
    "Retriever": {"selected": "...", "components": {...}},
    "Generator": {"selected": "...", "components": {...}}
}
```

Each component has:
- `name`: Component identifier
- `config`: Configurable parameters (model, size, overlap, etc.)
- `description`: Human-readable description
- `available`: Checks for required env vars/libraries

### Environment Variables
- **Weaviate:** `WEAVIATE_URL_VERBA`, `WEAVIATE_API_KEY_VERBA`
- **LLM Providers:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, etc.
- **Embedders:** `OLLAMA_URL`, `VOYAGE_API_KEY`, `UPSTAGE_API_KEY`, etc.
- **Readers:** `UNSTRUCTURED_API_KEY`, `ASSEMBLYAI_API_KEY`, `GITHUB_TOKEN`, etc.
- **Deployment:** `DEFAULT_DEPLOYMENT` (Local/Docker/Weaviate/Custom)
- **System Prompt:** `SYSYEM_MESSAGE_PROMPT` (default RAG prompt)

---

## Deployment Options

### 1. **Local Deployment**
- Uses Weaviate Embedded (runs in-process)
- Not supported on Windows (use Docker instead)
- No external Weaviate instance needed

### 2. **Docker Deployment**
- Separate Verba + Weaviate containers
- Managed via `docker-compose.yml`
- Verba on port 8000, Weaviate on port 8080
- Persistent volume for Weaviate data

### 3. **Cloud Deployment**
- Connects to Weaviate Cloud Services (WCS)
- Requires `WEAVIATE_URL_VERBA` and `WEAVIATE_API_KEY_VERBA`

### 4. **Custom Deployment**
- User-specified Weaviate instance (host, port, key)

---

## Advanced Features

### 1. **Hybrid Search**
Combines semantic (vector) and keyword (BM25) search with alpha=0.5 weighting.

### 2. **Context Windows**
Retrieves surrounding chunks for better context (configurable window size).

### 3. **3D Vector Visualization**
PCA reduces embeddings to 3D for visual exploration of document space.

### 4. **Autocomplete Suggestions**
Stores past queries for autocomplete (stored in `VERBA_SUGGESTIONS`).

### 5. **Multi-Model Support**
Each document can use different embedders - collections are model-specific.

### 6. **Async Processing**
Parallel document processing with `asyncio.gather()` for speed.

### 7. **Batched Embedding**
Embeddings processed in configurable batch sizes to respect rate limits.

### 8. **Overwrite Protection**
Checks for duplicate documents by name (optional overwrite).

---

## Client Management

### ClientManager (`verba_manager.py`)
- Manages WebSocket/HTTP client connections
- Connection pooling with credential hashing
- Automatic cleanup of inactive clients (10-minute timeout)
- Thread-safe with asyncio locks

### LoggerManager (`server/helpers.py`)
- Real-time progress reporting via WebSocket
- File-level status tracking (STARTING, LOADING, CHUNKING, EMBEDDING, INGESTING, DONE, ERROR)

### BatchManager (`server/helpers.py`)
- Handles large file uploads in chunks
- Merges batches once complete

---

## Testing

- Located in `goldenverba/tests/`
- Document model tests in `test_document.py`
- Additional tests needed (noted as TODO in TECHNICAL.md)

---

## Known Limitations

1. **Weaviate Embedded:** Not supported on Windows (experimental mode)
2. **Multi-user:** Not designed for concurrent multi-user access
3. **API Endpoints:** Internal APIs not designed for external consumption
4. **Custom JSON:** No support for custom JSON structures yet
5. **Community Project:** Not maintained with production-level urgency

---

## Extension Points

To add new components:

1. Create class inheriting from base interface (`Reader`, `Chunker`, etc.)
2. Implement required methods
3. Add to component list in `managers.py` (e.g., `readers`, `chunkers`)
4. Set `requires_env` and `requires_library` for availability checking

Example locations:
- New Reader: `components/reader/`
- New Chunker: `components/chunking/`
- New Embedder: `components/embedding/`
- New Generator: `components/generation/`

---

## Security

- **CORS:** Same-origin policy enforced (except `/api/health`)
- **API Keys:** Stored in environment variables or entered via UI
- **No Auth:** Anonymous access enabled (single-user design)

---

## Performance Optimizations

- **Async/Await:** Throughout for I/O-bound operations
- **Batch Processing:** Embeddings, database inserts
- **Connection Pooling:** Reuse Weaviate clients
- **Streaming:** LLM responses streamed token-by-token
- **PCA Caching:** 3D coordinates stored with chunks

---

## Docker Configuration

### docker-compose.yml
- **verba service:** FastAPI app (port 8000)
- **weaviate service:** Vector DB (port 8080)
- **Optional ollama service:** Local LLM hosting
- **Networking:** Custom `ollama-docker` network
- **Volumes:** `weaviate_data` for persistence, `./data` mounted to Verba

### Environment Variables
Set in `.env` file or directly in `docker-compose.yml`:
- LLM API keys
- Ollama configuration
- Document processing API keys

---

## Summary

Verba is a well-architected RAG application with:

- **Modular Design:** Pluggable components for each pipeline stage
- **Flexible Configuration:** Per-document RAG settings
- **Multiple Integrations:** 7+ LLM providers, 7+ embedding models
- **Modern Stack:** FastAPI + Next.js + Weaviate
- **Docker-First:** Production-ready containerized deployment
- **Community-Driven:** Open-source with active development

The codebase demonstrates clean separation of concerns, with clear interfaces between layers and extensive use of async patterns for performance. The component manager system makes it easy to extend with new models or processing strategies.
