"""HTTP client for communicating with model endpoints."""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import httpx
from httpx import AsyncClient, TimeoutException, HTTPStatusError

from schemas.generation import GenerationRequest, GenerationOutput
from schemas.judging import JudgingRequest, JudgingOutput
from schemas.chairman import ChairmanRequest, ChairmanOutput

logger = logging.getLogger(__name__)


class ModelClient:
    """Async HTTP client for model endpoints."""
    
    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize model client.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Initial retry delay in seconds (exponential backoff)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client: Optional[AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.client = AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
    
    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            url: URL to request
            **kwargs: Additional arguments for request
        
        Returns:
            Response object
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            
            except (TimeoutException, HTTPStatusError, httpx.RequestError) as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Request to {url} failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request to {url} failed after {self.max_retries} attempts: {e}")
        
        raise last_exception
    
    async def generate(
        self,
        url: str,
        request: GenerationRequest,
    ) -> Optional[GenerationOutput]:
        """
        Call generation endpoint.
        
        Args:
            url: Member endpoint URL
            request: Generation request
        
        Returns:
            Generation output or None if failed
        """
        try:
            response = await self._request_with_retry(
                "POST",
                f"{url}/generate",
                json=request.model_dump(),
            )
            
            data = response.json()
            return GenerationOutput(**data)
        
        except Exception as e:
            logger.error(f"Generation request to {url} failed: {e}")
            return None
    
    async def judge(
        self,
        url: str,
        request: JudgingRequest,
    ) -> Optional[JudgingOutput]:
        """
        Call judging endpoint.
        
        Args:
            url: Member endpoint URL
            request: Judging request
        
        Returns:
            Judging output or None if failed
        """
        try:
            response = await self._request_with_retry(
                "POST",
                f"{url}/judge",
                json=request.model_dump(),
            )
            
            data = response.json()
            return JudgingOutput(**data)
        
        except Exception as e:
            logger.error(f"Judging request to {url} failed: {e}")
            return None
    
    async def synthesize(
        self,
        url: str,
        request: ChairmanRequest,
    ) -> Optional[ChairmanOutput]:
        """
        Call chairman synthesis endpoint.
        
        Args:
            url: Chairman endpoint URL
            request: Chairman request
        
        Returns:
            Chairman output or None if failed
        """
        try:
            response = await self._request_with_retry(
                "POST",
                f"{url}/synthesize",
                json=request.model_dump(),
            )
            
            data = response.json()
            return ChairmanOutput(**data)
        
        except Exception as e:
            logger.error(f"Synthesis request to {url} failed: {e}")
            return None
    
    async def health_check(self, url: str) -> bool:
        """
        Check if endpoint is healthy.
        
        Args:
            url: Endpoint URL
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            return data.get("status") == "healthy"
        
        except Exception as e:
            logger.warning(f"Health check failed for {url}: {e}")
            return False


async def parallel_generate(
    client: ModelClient,
    member_urls: List[str],
    request: GenerationRequest,
) -> Dict[str, Optional[GenerationOutput]]:
    """
    Send generation request to multiple members in parallel.
    
    Args:
        client: Model client
        member_urls: List of member URLs
        request: Generation request
    
    Returns:
        Dictionary mapping member URL to output (None if failed)
    """
    tasks = [
        client.generate(url, request)
        for url in member_urls
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    output_map = {}
    for url, result in zip(member_urls, results):
        if isinstance(result, Exception):
            logger.error(f"Exception for {url}: {result}")
            output_map[url] = None
        else:
            output_map[url] = result
    
    return output_map


async def parallel_judge(
    client: ModelClient,
    member_urls: List[str],
    request: JudgingRequest,
) -> Dict[str, Optional[JudgingOutput]]:
    """
    Send judging request to multiple members in parallel.
    
    Args:
        client: Model client
        member_urls: List of member URLs
        request: Judging request
    
    Returns:
        Dictionary mapping member URL to output (None if failed)
    """
    tasks = [
        client.judge(url, request)
        for url in member_urls
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    output_map = {}
    for url, result in zip(member_urls, results):
        if isinstance(result, Exception):
            logger.error(f"Exception for {url}: {result}")
            output_map[url] = None
        else:
            output_map[url] = result
    
    return output_map


async def check_all_endpoints(
    client: ModelClient,
    urls: List[str]
) -> Dict[str, bool]:
    """
    Check health of all endpoints.
    
    Args:
        client: Model client
        urls: List of endpoint URLs
    
    Returns:
        Dictionary mapping URL to health status
    """
    tasks = [client.health_check(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    health_map = {}
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            health_map[url] = False
        else:
            health_map[url] = result
    
    return health_map

