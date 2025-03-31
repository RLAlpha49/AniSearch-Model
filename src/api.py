"""
# AniSearch API Server

A FastAPI server that exposes the AniSearch functionality through HTTP endpoints.

This module provides a REST API for searching anime and manga datasets using
cross-encoder models for semantic similarity. It allows clients to:

1. Search for anime matching a description
2. Search for manga matching a description
3. List available models
4. Get health check status

## Features

- **RESTful API**: Clean, standards-compliant API design
- **Interactive Documentation**: Automatic OpenAPI/Swagger UI at `/docs`
- **CORS Support**: Configurable cross-origin resource sharing
- **Multi-worker Architecture**: Handles concurrent requests efficiently
- **Model Caching**: Avoids reloading models for each request
- **Route Restrictions**: Configurable endpoint enabling/disabling for production
- **Custom Performance Settings**: Configurable worker count and connection limits

## API Endpoints

| Endpoint           | Method | Description                              |
|--------------------|--------|------------------------------------------|
| `/`                | GET    | Health check and CUDA availability       |
| `/models`          | GET    | List available models and fine-tuned models |
| `/search/anime`    | POST   | Search for anime matching a description  |
| `/search/manga`    | POST   | Search for manga matching a description  |

## Server Usage

```bash
# Basic usage
python -m src.api

# With custom settings
python -m src.api --host=127.0.0.1 --port=9000 --workers=4

# Production mode with restricted routes
python -m src.api --enable-routes=search --cors-origins="https://yourdomain.com"
```

## GPU Acceleration

For optimal performance, especially with larger models, using a GPU is recommended.
To enable GPU support, install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

You can then specify `device=cuda` in your API requests to utilize GPU acceleration.
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from src.utils.logging_config import setup_logging
from src.main import get_search_model
from src.models.base_search_model import BaseSearchModel
from src.training.utils import get_device

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="AniSearch API",
    description="API for searching anime and manga using semantic similarity",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for search models to avoid reloading
model_cache = {}


# Models for request/response data
class SearchRequest(BaseModel):
    """
    Request model for anime and manga search endpoints.

    This model defines the required and optional parameters for search requests.
    It includes validation rules to ensure the parameters are within acceptable
    ranges.

    Attributes:
        query: The search query text describing the anime/manga to find
        num_results: Number of results to return (default: 5)
        batch_size: Batch size for processing the search in the model (default: 32)

    Example:
        ```python
        search_request = SearchRequest(
            query="A story about a young wizard learning magic",
            num_results=10,
            batch_size=64
        )
        ```
    """

    query: str = Field(..., description="The search query text", min_length=1)
    num_results: int = Field(5, description="Number of results to return", ge=1, le=100)
    batch_size: int = Field(32, description="Batch size for processing", ge=8, le=512)


class SearchResult(BaseModel):
    """
    Individual search result item returned by the search endpoints.

    This model represents a single anime or manga entry matched by the search.
    It includes the basic information needed to display the result to the user.

    Attributes:
        id: Unique identifier for the entry (anime_id or manga_id)
        title: Title of the anime/manga
        score: Relevance score between 0.0 and 1.0 (higher is more relevant)
        synopsis: Partial synopsis text (may be truncated for display)

    Example:
        ```python
        result = SearchResult(
            id=1535,
            title="Death Note",
            score=0.92,
            synopsis="Light Yagami is a genius high schooler who discovers..."
        )
        ```
    """

    id: Union[int, str] = Field(..., description="Unique identifier for the entry")
    title: str = Field(..., description="Title of the anime/manga")
    score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)
    synopsis: str = Field(..., description="Synopsis text (may be truncated)")


class SearchResponse(BaseModel):
    """
    Response model for anime and manga search endpoints.

    This model defines the structure of the response returned by the search
    endpoints. It includes the search results, execution time, and device used
    for computation.

    Attributes:
        results: List of search results sorted by relevance
        execution_time_ms: Total execution time of the search in milliseconds
        device_used: The device used for computation (e.g., 'cpu', 'cuda')

    Example:
        ```python
        response = SearchResponse(
            results=[
                SearchResult(id=1535, title="Death Note", score=0.92, synopsis="..."),
                SearchResult(id=5114, title="Fullmetal Alchemist", score=0.85, synopsis="...")
            ],
            execution_time_ms=156.32,
            device_used="cuda"
        )
        ```
    """

    results: List[SearchResult] = Field(..., description="Search results")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    device_used: str = Field(..., description="Device used for computation (CPU/CUDA)")


class HealthResponse(BaseModel):
    """
    Response model for the health check endpoint.

    This model defines the structure of the response returned by the health
    check endpoint. It includes the overall API status, the status of each
    model type, and information about CUDA availability.

    Attributes:
        status: Overall status of the API ('healthy' or 'degraded')
        models_loaded: Dictionary of model types and their loading status
        cuda_available: Whether CUDA is available on the system

    Example:
        ```python
        health = HealthResponse(
            status="healthy",
            models_loaded={"anime": True, "manga": True},
            cuda_available=True
        )
        ```
    """

    status: str = Field(..., description="Health status of the API")
    models_loaded: Dict[str, bool] = Field(
        ..., description="Status of the search models"
    )
    cuda_available: bool = Field(
        ..., description="Whether CUDA is available on this system"
    )


class ModelsResponse(BaseModel):
    """
    Response model for the models endpoint.

    This model defines the structure of the response returned by the models
    endpoint. It includes information about available pre-trained models
    and any fine-tuned models.

    Attributes:
        models: Dictionary of model categories and available models
        fine_tuned: Dictionary of fine-tuned model names and their paths

    Example:
        ```python
        models = ModelsResponse(
            models={
                "Semantic Search": {
                    "cross-encoder/ms-marco-MiniLM-L-6-v2": "Recommended for general search"
                }
            },
            fine_tuned={
                "anime-v1": "model/fine-tuned/anime-v1"
            }
        )
        ```
    """

    models: Dict[str, Dict[str, str]] = Field(
        ..., description="Available models by category"
    )
    fine_tuned: Dict[str, str] = Field(..., description="Available fine-tuned models")


def get_or_create_model(
    dataset_type: str,
    model_name: str,
    device: Optional[str] = None,
    include_light_novels: bool = False,
) -> BaseSearchModel:
    """
    Get a cached model or create a new one if not already cached.

    This function manages the model cache to avoid reloading models for each request.
    It handles device selection, CUDA availability checking, and model initialization.

    Args:
        dataset_type: The type of dataset to use ('anime' or 'manga')
        model_name: The name or path of the model to use
        device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.)
            If None, automatically selects the best available device
        include_light_novels: Whether to include light novels in manga search results
            Only relevant for manga dataset_type

    Returns:
        BaseSearchModel: An initialized search model ready for queries

    Raises:
        ValueError: If the model or dataset cannot be loaded
        RuntimeError: If there are issues initializing the model

    Note:
        If CUDA is requested but not available, it will automatically
        fall back to CPU with a warning.
    """
    # Check if CUDA is available when 'cuda' is requested
    import torch  # pylint: disable=import-outside-toplevel

    cuda_requested = device is not None and "cuda" in device
    cuda_available = torch.cuda.is_available()

    # Force CPU if CUDA is requested but not available
    if cuda_requested and not cuda_available:
        logger.warning("CUDA was requested but is not available. Falling back to CPU.")
        selected_device = "cpu"
    else:
        # Use the specified device or auto-detect
        selected_device = get_device(device)

    # Create a unique key for this configuration
    key = f"{dataset_type}_{model_name}_{selected_device}_{include_light_novels}"

    if key not in model_cache:
        logger.info("Creating new model: %s on device: %s", key, selected_device)
        model = get_search_model(
            dataset_type=dataset_type,
            model_name=model_name,
            device=selected_device,
            include_light_novels=include_light_novels,
        )

        # The model's device is already set in its constructor
        model_cache[key] = model

    return model_cache[key]


@app.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check if the API is running and ready to handle requests.

    This endpoint verifies that the API server is operational and provides
    information about the status of different components:

    1. Whether the API server itself is running
    2. Whether each model type (anime, manga) can be loaded
    3. Whether CUDA is available for GPU acceleration

    Returns:
        HealthResponse: The health status of the API

        - **status**: "healthy" if critical components are working, "degraded" otherwise
        - **models_loaded**: Dictionary indicating which models loaded successfully
        - **cuda_available**: Boolean indicating if CUDA is available for GPU acceleration

    Example:
        ```bash
        curl -X GET "http://localhost:8000/"
        ```

    Note:
        This endpoint intentionally uses CPU for model loading checks to avoid
        GPU memory issues during health checking.
    """
    # Check CUDA availability
    import torch  # pylint: disable=import-outside-toplevel

    cuda_available = torch.cuda.is_available()

    # Check if models can be loaded
    models_loaded = {
        "anime": False,
        "manga": False,
    }

    try:
        # Try on CPU to avoid GPU memory issues during health check
        get_or_create_model(
            "anime", "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )
        models_loaded["anime"] = True
    except (ImportError, ValueError, RuntimeError, FileNotFoundError) as e:
        logger.error("Error loading anime model: %s", str(e))

    try:
        # Try on CPU to avoid GPU memory issues during health check
        get_or_create_model(
            "manga", "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )
        models_loaded["manga"] = True
    except (ImportError, ValueError, RuntimeError, FileNotFoundError) as e:
        logger.error("Error loading manga model: %s", str(e))

    return HealthResponse(
        status="healthy" if any(models_loaded.values()) else "degraded",
        models_loaded=models_loaded,
        cuda_available=cuda_available,
    )


@app.get("/models", response_model=ModelsResponse)
async def get_available_models() -> ModelsResponse:
    """
    Get a list of available pre-trained and fine-tuned models.

    This endpoint returns information about models that can be used with the
    search endpoints. It includes:

    1. Pre-trained models categorized by type (e.g., Semantic Search, Question Answering)
    2. Fine-tuned models specifically trained for anime/manga search

    Returns:
        ModelsResponse: Available models and their descriptions

        - **models**: Dictionary of model categories and available pre-trained models
        - **fine_tuned**: Dictionary of fine-tuned model names and their paths

    Example:
        ```bash
        curl -X GET "http://localhost:8000/models"
        ```

    Note:
        Fine-tuned models are located in the `model/fine-tuned` directory.
        The API will only list models that have a valid configuration file.
    """
    # Convert Mapping to Dict to satisfy the type checker
    models = dict(BaseSearchModel.list_available_models())
    fine_tuned = BaseSearchModel.list_fine_tuned_models()

    return ModelsResponse(models=models, fine_tuned=fine_tuned)


@app.post("/search/anime", response_model=SearchResponse)
async def search_anime(
    request: SearchRequest,
    model_name: str = Query(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", description="Model name or path"
    ),
    device: Optional[str] = Query(
        None,
        description="Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.). "
        "If not specified, uses the best available device.",
    ),
) -> SearchResponse:
    """
    Search for anime matching the provided description.
    
    This endpoint performs semantic search against the anime dataset using
    the specified model, returning the most relevant matches sorted by score.
    
    ## Parameters
    
    - **request**: The search request body containing:
        - **query**: The search query text describing the anime
        - **num_results**: Number of results to return (default: 5, max: 100)
        - **batch_size**: Batch size for processing (default: 32)
    
    - **model_name**: The model to use for search (query parameter)
        - Can be a pre-trained model name or path to a fine-tuned model
        - Default: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    - **device**: The device to run the model on (query parameter)
        - Options: 'cpu', 'cuda', 'cuda:0', etc.
        - If not specified, uses the best available device
    
    ## Returns
    
    - **results**: List of anime matching the query, sorted by relevance
    - **execution_time_ms**: Time taken to execute the search in milliseconds
    - **device_used**: The device used for computation (e.g., 'cpu', 'cuda')
    
    ## Example
    
    ```bash
    curl -X POST "http://localhost:8000/search/anime?model_name=cross-encoder/ms-marco-MiniLM-L-6-v2&device=cuda" \\
      -H "Content-Type: application/json" \\
      -d '{"query": "A story about robots and AI"}'
    ```
    
    ## Notes
    
    - For optimal performance on large queries, use GPU acceleration with `device=cuda`
    - Model caching is used to avoid reloading models between requests
    - Results include truncated synopses; full content is available in the dataset
    """
    import time  # pylint: disable=import-outside-toplevel

    try:
        # Get the search model
        start_time = time.time()
        search_model = get_or_create_model("anime", model_name, device=device)

        # Perform the search
        results = search_model.search(
            query=request.query,
            num_results=request.num_results,
            batch_size=request.batch_size,
        )

        # Convert to response format
        execution_time_ms = (time.time() - start_time) * 1000
        return SearchResponse(
            results=[SearchResult(**result) for result in results],
            execution_time_ms=execution_time_ms,
            device_used=search_model.device,
        )
    except (ImportError, ValueError, RuntimeError, FileNotFoundError) as e:
        logger.error("Error in anime search: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error performing search: {str(e)}"
        ) from e


@app.post("/search/manga", response_model=SearchResponse)
async def search_manga(
    request: SearchRequest,
    model_name: str = Query(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", description="Model name or path"
    ),
    include_light_novels: bool = Query(
        False, description="Whether to include light novels in search results"
    ),
    device: Optional[str] = Query(
        None,
        description="Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.). "
        "If not specified, uses the best available device.",
    ),
) -> SearchResponse:
    """
    Search for manga matching the provided description.
    
    This endpoint performs semantic search against the manga dataset using
    the specified model, returning the most relevant matches sorted by score.
    
    ## Parameters
    
    - **request**: The search request body containing:
        - **query**: The search query text describing the manga
        - **num_results**: Number of results to return (default: 5, max: 100)
        - **batch_size**: Batch size for processing (default: 32)
    
    - **model_name**: The model to use for search (query parameter)
        - Can be a pre-trained model name or path to a fine-tuned model
        - Default: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    - **include_light_novels**: Whether to include light novels in results (query parameter)
        - Default: false
    
    - **device**: The device to run the model on (query parameter)
        - Options: 'cpu', 'cuda', 'cuda:0', etc.
        - If not specified, uses the best available device
    
    ## Returns
    
    - **results**: List of manga matching the query, sorted by relevance
    - **execution_time_ms**: Time taken to execute the search in milliseconds
    - **device_used**: The device used for computation (e.g., 'cpu', 'cuda')
    
    ## Example
    
    ```bash
    curl -X POST "http://localhost:8000/search/manga?include_light_novels=true&device=cuda" \\
      -H "Content-Type: application/json" \\
      -d '{"query": "A fantasy adventure in a magical world", "num_results": 10}'
    ```
    
    ## Notes
    
    - Use `include_light_novels=true` to include light novels in search results
    - For optimal performance on large queries, use GPU acceleration with `device=cuda`
    - Model caching is used to avoid reloading models between requests
    - Results include truncated synopses; full content is available in the dataset
    """
    import time  # pylint: disable=import-outside-toplevel

    try:
        # Get the search model
        start_time = time.time()
        search_model = get_or_create_model(
            "manga",
            model_name,
            device=device,
            include_light_novels=include_light_novels,
        )

        # Perform the search
        results = search_model.search(
            query=request.query,
            num_results=request.num_results,
            batch_size=request.batch_size,
        )

        # Convert to response format
        execution_time_ms = (time.time() - start_time) * 1000
        return SearchResponse(
            results=[SearchResult(**result) for result in results],
            execution_time_ms=execution_time_ms,
            device_used=search_model.device,
        )
    except (ImportError, ValueError, RuntimeError, FileNotFoundError) as e:
        logger.error("Error in manga search: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error performing search: {str(e)}"
        ) from e


if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    import argparse
    import tempfile

    # Setup command line arguments
    parser = argparse.ArgumentParser(description="AniSearch API Server")

    # Server configuration
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )

    # CORS configuration
    parser.add_argument(
        "--cors-origins",
        type=str,
        default="*",
        help="Comma-separated list of allowed origins for CORS (default: '*')",
    )
    parser.add_argument(
        "--cors-methods",
        type=str,
        default="*",
        help="Comma-separated list of allowed HTTP methods for CORS (default: '*')",
    )
    parser.add_argument(
        "--cors-headers",
        type=str,
        default="*",
        help="Comma-separated list of allowed HTTP headers for CORS (default: '*')",
    )

    # Performance configuration
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: half of CPU cores)",
    )
    parser.add_argument(
        "--limit-concurrency",
        type=int,
        default=50,
        help="Maximum number of concurrent connections (default: 50)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout for keep-alive connections in seconds (default: 30)",
    )

    # Route restrictions
    parser.add_argument(
        "--enable-routes",
        type=str,
        default="all",
        help=(
            "Comma-separated list of routes to enable. Options: 'all', 'search', "
            "'health', 'models'. Use 'search' for production to only enable search "
            "endpoints (default: 'all')"
        ),
    )

    args = parser.parse_args()

    # Process CORS arguments
    # Convert comma-separated strings to lists (or keep as "*" if wildcard)
    origins = (
        [origin.strip() for origin in args.cors_origins.split(",")]
        if args.cors_origins != "*"
        else ["*"]
    )
    methods = (
        [method.strip() for method in args.cors_methods.split(",")]
        if args.cors_methods != "*"
        else ["*"]
    )
    headers = (
        [header.strip() for header in args.cors_headers.split(",")]
        if args.cors_headers != "*"
        else ["*"]
    )

    # Determine the number of workers
    if args.workers is None:
        # Use half the cores rounded down with a minimum of 1 to avoid overloading the system
        # Since each worker will also use a model that might require significant resources
        num_workers = max(1, int(multiprocessing.cpu_count() / 2))
    else:
        num_workers = args.workers

    # Process route restrictions
    enabled_routes = [route.strip().lower() for route in args.enable_routes.split(",")]
    all_routes_enabled = "all" in enabled_routes

    # For default configuration (all routes), use the module directly
    if all_routes_enabled:
        logger.info(
            "Starting AniSearch API server with %d workers and all routes enabled",
            num_workers,
        )
        logger.info(
            "CORS configuration: origins=%s, methods=%s, headers=%s",
            origins,
            methods,
            headers,
        )

        # Update CORS middleware with command line arguments
        app.middleware_stack = None  # Clear existing middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=methods,
            allow_headers=headers,
        )

        # Use the module import string
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            workers=num_workers,
            limit_concurrency=args.limit_concurrency,
            timeout_keep_alive=args.timeout,
        )
    else:
        # For restricted routes, create a temporary module to allow multiple workers
        # Create a new FastAPI app with only the enabled routes
        restricted_app = FastAPI(
            title=app.title,
            description=app.description,
            version=app.version,
        )

        # Add CORS middleware to the new app
        restricted_app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=methods,
            allow_headers=headers,
        )

        # Add only the enabled routes
        if "health" in enabled_routes:
            logger.info("Enabling health check endpoint: /")

            @restricted_app.get("/", response_model=HealthResponse)
            async def restricted_health_check():
                """
                Health check endpoint for the restricted API mode.

                This endpoint verifies that the API server is operational in restricted mode
                and provides information about the status of different components.

                Returns:
                    HealthResponse: The health status of the API
                """
                return await health_check()

        if "models" in enabled_routes:
            logger.info("Enabling models endpoint: /models")

            @restricted_app.get("/models", response_model=ModelsResponse)
            async def restricted_get_models():
                """
                List available models endpoint for the restricted API mode.

                This endpoint returns information about models that can be used with
                the search endpoints in restricted mode.

                Returns:
                    ModelsResponse: Available models and their descriptions
                """
                return await get_available_models()

        if "search" in enabled_routes:
            logger.info("Enabling search endpoints: /search/anime and /search/manga")

            # Add the anime search endpoint
            @restricted_app.post("/search/anime", response_model=SearchResponse)
            async def restricted_search_anime(*args, **kwargs):
                """
                Search for anime endpoint for the restricted API mode.

                This endpoint performs semantic search against the anime dataset
                using the specified model in restricted mode.

                Parameters are the same as the regular search_anime endpoint.

                Returns:
                    SearchResponse: The search results with relevant anime matches
                """
                return await search_anime(*args, **kwargs)

            # Add the manga search endpoint
            @restricted_app.post("/search/manga", response_model=SearchResponse)
            async def restricted_search_manga(*args, **kwargs):
                """
                Search for manga endpoint for the restricted API mode.

                This endpoint performs semantic search against the manga dataset
                using the specified model in restricted mode.

                Parameters are the same as the regular search_manga endpoint.

                Returns:
                    SearchResponse: The search results with relevant manga matches
                """
                return await search_manga(*args, **kwargs)

        # For multiple workers with restricted routes, create a temp module file
        if num_workers > 1:
            # Create a temporary directory to hold our module
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a new Python file with our app
                temp_module_path = os.path.join(temp_dir, "temp_api.py")

                # Make the directory importable
                with open(
                    os.path.join(temp_dir, "__init__.py"), "w", encoding="utf-8"
                ) as f:
                    f.write("")

                # Write the module code
                with open(temp_module_path, "w", encoding="utf-8") as f:
                    # Import necessary dependencies
                    f.write("from fastapi import FastAPI\n")
                    f.write("from fastapi.middleware.cors import CORSMiddleware\n\n")
                    f.write("# The restricted FastAPI app\n")
                    f.write("app = None\n\n")
                    f.write("def init_app():\n")
                    f.write('    """\n')
                    f.write(
                        "    Initialize the FastAPI application for the restricted routes mode.\n"
                    )
                    f.write("    \n")
                    f.write(
                        "    This function is called once when the module is loaded to create and\n"
                    )
                    f.write(
                        "    configure the FastAPI application with only the enabled routes.\n"
                    )
                    f.write(
                        "    It handles importing necessary dependencies, setting up CORS,\n"
                    )
                    f.write("    and registering the route handlers.\n")
                    f.write('    """\n')
                    f.write("    global app\n")
                    f.write("    # Create the app only once\n")
                    f.write("    if app is None:\n")
                    f.write(
                        "        # Import all dependencies needed by the app here\n"
                    )
                    f.write("        import sys\n")
                    # Fix path escaping by replacing backslashes with forward slashes
                    safe_path = os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    ).replace("\\", "/")
                    f.write(f"        sys.path.append('{safe_path}')\n")
                    f.write(
                        "        from src.utils.logging_config import setup_logging\n"
                    )
                    f.write(
                        "        from src.api import SearchResponse, search_anime, search_manga\n"
                    )
                    f.write(
                        "        from src.api import HealthResponse, health_check\n"
                    )
                    f.write(
                        "        from src.api import ModelsResponse, get_available_models\n\n"
                    )
                    f.write("        # Configure logging\n")
                    f.write("        setup_logging()\n\n")
                    f.write("        # Create the app\n")
                    f.write(
                        f"        app = FastAPI(title='{app.title}', description='{app.description}', "
                        f"version='{app.version}')\n\n"
                    )
                    f.write("        # Add CORS middleware\n")
                    f.write("        app.add_middleware(\n")
                    f.write("            CORSMiddleware,\n")
                    f.write(f"            allow_origins={origins},\n")
                    f.write("            allow_credentials=True,\n")
                    f.write(f"            allow_methods={methods},\n")
                    f.write(f"            allow_headers={headers},\n")
                    f.write("        )\n\n")

                    # Add enabled routes
                    if "health" in enabled_routes:
                        f.write("        # Health check endpoint\n")
                        f.write(
                            "        @app.get('/', response_model=HealthResponse)\n"
                        )
                        f.write("        async def restricted_health_check():\n")
                        f.write('            """\n')
                        f.write(
                            "            Health check endpoint for the restricted API mode.\n"
                        )
                        f.write("            \n")
                        f.write(
                            "            This endpoint verifies that the API server is operational in restricted mode\n"
                        )
                        f.write(
                            "            and provides information about the status of different components.\n"
                        )
                        f.write("            \n")
                        f.write("            Returns:\n")
                        f.write(
                            "                HealthResponse: The health status of the API\n"
                        )
                        f.write('            """\n')
                        f.write("            return await health_check()\n\n")

                    if "models" in enabled_routes:
                        f.write("        # Models endpoint\n")
                        f.write(
                            "        @app.get('/models', response_model=ModelsResponse)\n"
                        )
                        f.write("        async def restricted_get_models():\n")
                        f.write('            """\n')
                        f.write(
                            "            List available models endpoint for the restricted API mode.\n"
                        )
                        f.write("            \n")
                        f.write(
                            "            This endpoint returns information about models that can be used with\n"
                        )
                        f.write(
                            "            the search endpoints in restricted mode.\n"
                        )
                        f.write("            \n")
                        f.write("            Returns:\n")
                        f.write(
                            "                ModelsResponse: Available models and their descriptions\n"
                        )
                        f.write('            """\n')
                        f.write("            return await get_available_models()\n\n")

                    if "search" in enabled_routes:
                        f.write("        # Search endpoints\n")
                        f.write(
                            "        @app.post('/search/anime', response_model=SearchResponse)\n"
                        )
                        f.write(
                            "        async def restricted_search_anime(*args, **kwargs):\n"
                        )
                        f.write('            """\n')
                        f.write(
                            "            Search for anime endpoint for the restricted API mode.\n"
                        )
                        f.write("            \n")
                        f.write(
                            "            This endpoint performs semantic search against the anime dataset\n"
                        )
                        f.write(
                            "            using the specified model in restricted mode.\n"
                        )
                        f.write("            \n")
                        f.write(
                            "            Parameters are the same as the regular search_anime endpoint.\n"
                        )
                        f.write("            \n")
                        f.write("            Returns:\n")
                        f.write(
                            "                SearchResponse: The search results with relevant anime matches\n"
                        )
                        f.write('            """\n')
                        f.write(
                            "            return await search_anime(*args, **kwargs)\n\n"
                        )

                        f.write(
                            "        @app.post('/search/manga', response_model=SearchResponse)\n"
                        )
                        f.write(
                            "        async def restricted_search_manga(*args, **kwargs):\n"
                        )
                        f.write('            """\n')
                        f.write(
                            "            Search for manga endpoint for the restricted API mode.\n"
                        )
                        f.write("            \n")
                        f.write(
                            "            This endpoint performs semantic search against the manga dataset\n"
                        )
                        f.write(
                            "            using the specified model in restricted mode.\n"
                        )
                        f.write("            \n")
                        f.write(
                            "            Parameters are the same as the regular search_manga endpoint.\n"
                        )
                        f.write("            \n")
                        f.write("            Returns:\n")
                        f.write(
                            "                SearchResponse: The search results with relevant manga matches\n"
                        )
                        f.write('            """\n')
                        f.write(
                            "            return await search_manga(*args, **kwargs)\n\n"
                        )

                # Add the app initialization
                with open(
                    os.path.join(temp_dir, "temp_api.py"), "a", encoding="utf-8"
                ) as f:
                    f.write("\n# Initialize the app\n")
                    f.write("init_app()\n")

                # Add the sys.path so Python can find our temporary module
                sys.path.insert(0, temp_dir)

                # Run uvicorn with the temporary module
                logger.info(
                    "Starting AniSearch API server with %d workers and restricted routes",
                    num_workers,
                )
                logger.info(
                    "CORS configuration: origins=%s, methods=%s, headers=%s",
                    origins,
                    methods,
                    headers,
                )
                logger.info("Route configuration: %s", args.enable_routes)

                try:
                    uvicorn.run(
                        "temp_api:app",
                        host=args.host,
                        port=args.port,
                        workers=num_workers,
                        limit_concurrency=args.limit_concurrency,
                        timeout_keep_alive=args.timeout,
                    )
                finally:
                    # Clean up by removing the temp directory from sys.path
                    if temp_dir in sys.path:
                        sys.path.remove(temp_dir)
        else:
            # For single worker, just use the app directly
            logger.info(
                "Starting AniSearch API server with 1 worker and restricted routes"
            )
            logger.info(
                "CORS configuration: origins=%s, methods=%s, headers=%s",
                origins,
                methods,
                headers,
            )
            logger.info("Route configuration: %s", args.enable_routes)

            uvicorn.run(
                restricted_app,
                host=args.host,
                port=args.port,
                workers=1,
                limit_concurrency=args.limit_concurrency,
                timeout_keep_alive=args.timeout,
            )
