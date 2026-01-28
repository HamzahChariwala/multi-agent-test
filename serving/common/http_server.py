"""FastAPI base classes for model serving endpoints."""

from typing import Optional, Callable, Any
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from schemas.generation import GenerationRequest, GenerationOutput
from schemas.judging import JudgingRequest, JudgingOutput


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gpu_id: str
    model_loaded: bool


class BaseModelServer:
    """Base class for model serving endpoints."""
    
    def __init__(
        self,
        gpu_id: str,
        port: int,
        model_name: Optional[str] = None,
    ):
        """
        Initialize base model server.
        
        Args:
            gpu_id: GPU identifier (e.g., 't4_gpu1')
            port: Port to serve on
            model_name: Optional model name for identification
        """
        self.gpu_id = gpu_id
        self.port = port
        self.model_name = model_name or "unknown"
        self.model_loaded = False
        
        # Create FastAPI app
        self.app = FastAPI(
            title=f"Model Server - {gpu_id}",
            description=f"Model serving endpoint for {gpu_id}",
            version="1.0.0",
        )
        
        # Register routes
        self._register_routes()
        
        # Register exception handlers
        self._register_exception_handlers()
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy" if self.model_loaded else "starting",
                gpu_id=self.gpu_id,
                model_loaded=self.model_loaded,
            )
        
        @self.app.post("/generate", response_model=GenerationOutput)
        async def generate(request: GenerationRequest):
            """Generation endpoint."""
            try:
                logger.info(f"[{self.gpu_id}] Received generation request: {request.request_id}")
                output = await self._handle_generate(request)
                logger.info(f"[{self.gpu_id}] Completed generation request: {request.request_id}")
                return output
            except Exception as e:
                logger.error(f"[{self.gpu_id}] Generation error: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/judge", response_model=JudgingOutput)
        async def judge(request: JudgingRequest):
            """Judging endpoint."""
            try:
                logger.info(f"[{self.gpu_id}] Received judging request: {request.request_id}")
                output = await self._handle_judge(request)
                logger.info(f"[{self.gpu_id}] Completed judging request: {request.request_id}")
                return output
            except Exception as e:
                logger.error(f"[{self.gpu_id}] Judging error: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "service": "Model Server",
                "gpu_id": self.gpu_id,
                "model": self.model_name,
                "endpoints": ["/health", "/generate", "/judge"],
            }
    
    def _register_exception_handlers(self):
        """Register exception handlers."""
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": f"Internal server error: {str(exc)}"}
            )
    
    async def _handle_generate(self, request: GenerationRequest) -> GenerationOutput:
        """
        Handle generation request. Must be implemented by subclass.
        
        Args:
            request: Generation request
        
        Returns:
            Generation output
        """
        raise NotImplementedError("Subclass must implement _handle_generate")
    
    async def _handle_judge(self, request: JudgingRequest) -> JudgingOutput:
        """
        Handle judging request. Must be implemented by subclass.
        
        Args:
            request: Judging request
        
        Returns:
            Judging output
        """
        raise NotImplementedError("Subclass must implement _handle_judge")
    
    def run(self, host: str = "0.0.0.0"):
        """
        Run the server.
        
        Args:
            host: Host to bind to
        """
        logger.info(f"Starting server for {self.gpu_id} on {host}:{self.port}")
        uvicorn.run(
            self.app,
            host=host,
            port=self.port,
            log_level="info",
        )
    
    async def startup(self):
        """Startup hook for initialization."""
        pass
    
    async def shutdown(self):
        """Shutdown hook for cleanup."""
        pass


class ChairmanServer:
    """Base class for chairman serving endpoint."""
    
    def __init__(self, port: int, model_name: Optional[str] = None):
        """
        Initialize chairman server.
        
        Args:
            port: Port to serve on
            model_name: Optional model name
        """
        self.port = port
        self.model_name = model_name or "chairman"
        self.model_loaded = False
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Chairman Server",
            description="Chairman synthesis endpoint",
            version="1.0.0",
        )
        
        # Register routes
        self._register_routes()
        
        # Register exception handlers
        self._register_exception_handlers()
    
    def _register_routes(self):
        """Register API routes."""
        from schemas.chairman import ChairmanRequest, ChairmanOutput
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy" if self.model_loaded else "starting",
                gpu_id="chairman",
                model_loaded=self.model_loaded,
            )
        
        @self.app.post("/synthesize", response_model=ChairmanOutput)
        async def synthesize(request: ChairmanRequest):
            """Chairman synthesis endpoint."""
            try:
                logger.info(f"[Chairman] Received synthesis request: {request.request_id}")
                output = await self._handle_synthesize(request)
                logger.info(f"[Chairman] Completed synthesis request: {request.request_id}")
                return output
            except Exception as e:
                logger.error(f"[Chairman] Synthesis error: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "service": "Chairman Server",
                "model": self.model_name,
                "endpoints": ["/health", "/synthesize"],
            }
    
    def _register_exception_handlers(self):
        """Register exception handlers."""
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": f"Internal server error: {str(exc)}"}
            )
    
    async def _handle_synthesize(self, request) -> Any:
        """
        Handle synthesis request. Must be implemented by subclass.
        
        Args:
            request: Chairman request
        
        Returns:
            Chairman output
        """
        raise NotImplementedError("Subclass must implement _handle_synthesize")
    
    def run(self, host: str = "0.0.0.0"):
        """
        Run the server.
        
        Args:
            host: Host to bind to
        """
        logger.info(f"Starting Chairman server on {host}:{self.port}")
        uvicorn.run(
            self.app,
            host=host,
            port=self.port,
            log_level="info",
        )

