"""
A100 Synthesis Server with 4-way Tensor Parallelism and KV Cache Reuse.

This server implements a two-phase workflow with NCCL broadcast coordination:
- Phase 1: Prefill initial context and store KV cache (parallel with T4 generation)
- Phase 2: Append council results and generate synthesis (reuse KV cache)

Architecture:
- 4 worker processes (one per GPU)
- Manual tensor parallelism with NCCL
- NCCL broadcast for rank coordination
- Single concurrent session (simplified state management)
- 5-minute KV cache timeout
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import time
import logging
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn

# Import common utilities
from serving.common.tp_utils import (
    shard_model_weights,
    all_reduce_logits,
    validate_kv_cache,
    estimate_kv_cache_size,
    synchronize_ranks,
    get_tp_group_info
)
from serving.common.profiling import ProfilerContext, profile_operation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Work Coordination
# ============================================================================

class WorkType(Enum):
    """Types of work that can be broadcast to all ranks."""
    SHUTDOWN = 0
    PREFILL_INITIAL = 1
    SYNTHESIZE_FINAL = 2
    SIMPLE_SYNTHESIS = 3  # Single-phase: prefill + decode


@dataclass
class WorkSignal:
    """Signal broadcast from rank 0 to coordinate work across all ranks."""
    work_type: int  # WorkType enum value
    request_id: str = ""
    context: str = ""
    appended_text: str = ""
    prompt: str = ""  # For simple synthesis
    max_tokens: int = 500
    temperature: float = 0.7
    metadata: Optional[Dict] = None


# ============================================================================
# Request/Response Models
# ============================================================================

class InitialPrefillRequest(BaseModel):
    """Request for Phase 1: Initial context prefill."""
    request_id: str
    context: str
    metadata: Optional[Dict[str, Any]] = None


class InitialPrefillResponse(BaseModel):
    """Response for Phase 1."""
    status: str
    request_id: str
    cache_stored: bool
    context_tokens: int
    prefill_time_ms: float
    cache_size_mb: float


class FinalSynthesisRequest(BaseModel):
    """Request for Phase 2: Final synthesis with KV cache reuse."""
    request_id: str
    appended_text: str
    max_tokens: int = 500
    temperature: float = 0.7
    metadata: Optional[Dict[str, Any]] = None


class SimpleSynthesisRequest(BaseModel):
    """Request for simple single-phase synthesis (no KV cache reuse)."""
    request_id: str
    prompt: str
    max_tokens: int = 500
    temperature: float = 0.7
    metadata: Optional[Dict[str, Any]] = None


class SimpleSynthesisResponse(BaseModel):
    """Response for simple synthesis."""
    status: str
    request_id: str
    text: str
    num_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float


class FinalSynthesisResponse(BaseModel):
    """Response for Phase 2."""
    status: str
    request_id: str
    synthesis: str
    num_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float


@dataclass
class CacheEntry:
    """Entry in KV cache store."""
    kv_cache: Tuple
    timestamp: float
    seq_len: int
    context_preview: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Synthesis Worker (One per GPU/Rank)
# ============================================================================

class SynthesisWorker:
    """
    Worker process for tensor-parallel synthesis with NCCL coordination.
    
    Each worker runs on one GPU and handles 1/4 of the model parameters.
    Workers coordinate via NCCL broadcast from rank 0.
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int = 4,
        model_name: str = "meta-llama/Llama-2-70b-chat-hf",
        precision: str = "bf16",
        port: int = 8020,
        profiling_enabled: bool = False,
        cache_timeout: int = 300  # 5 minutes
    ):
        self.rank = rank
        self.world_size = world_size
        self.model_name = model_name
        self.precision = precision
        self.port = port if rank == 0 else None  # Only rank 0 runs HTTP server
        self.profiling_enabled = profiling_enabled
        self.cache_timeout = cache_timeout
        
        self.device = f"cuda:{rank}"
        self.model = None
        self.tokenizer = None
        
        # KV cache store (shared across all ranks in memory)
        self.kv_cache_store: Dict[str, CacheEntry] = {}
        
        # Session management
        self.active_session_id: Optional[str] = None
        
        # Work coordination
        self.running = True
        self.pending_response = None  # Stores response from non-rank-0 processes
        
        logger.info(f"[Rank {rank}] Initializing SynthesisWorker")
    
    def initialize(self):
        """Initialize model and tensor parallel group."""
        logger.info(f"[Rank {self.rank}] Loading model {self.model_name}")
        
        # Load tokenizer (all ranks)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        
        logger.info(f"[Rank {self.rank}] Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # IMPORTANT: Shard weights ON CPU first, BEFORE moving to GPU
        # This prevents loading full 70B model (79GB) on each GPU
        logger.info(f"[Rank {self.rank}] Sharding weights on CPU...")
        num_sharded = shard_model_weights(
            self.model,
            rank=self.rank,
            world_size=self.world_size,
            verbose=(self.rank == 0)
        )
        
        # Now move SHARDED model to GPU (~17.5GB per GPU for 70B/4)
        logger.info(f"[Rank {self.rank}] Moving sharded model to {self.device}...")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"[Rank {self.rank}] Model loaded successfully ({num_sharded} params sharded)")
        
        # Synchronize all ranks
        synchronize_ranks()
        logger.info(f"[Rank {self.rank}] Initialization complete, ready for work")
    
    # ========================================================================
    # Work Coordination
    # ========================================================================
    
    def broadcast_work(self, signal: WorkSignal):
        """Broadcast work signal from rank 0 to all ranks."""
        if self.rank != 0:
            raise RuntimeError("Only rank 0 can broadcast work")
        
        # Create tensors for broadcasting
        work_type_tensor = torch.tensor([signal.work_type], dtype=torch.long, device=self.device)
        
        # Broadcast work type
        dist.broadcast(work_type_tensor, src=0)
        
        # Broadcast string data (request_id, context, etc.) as a pickled object
        # This is a simplified approach; production would use more efficient serialization
        data = {
            'request_id': signal.request_id,
            'context': signal.context,
            'appended_text': signal.appended_text,
            'prompt': signal.prompt,  # For simple synthesis
            'max_tokens': signal.max_tokens,
            'temperature': signal.temperature,
            'metadata': signal.metadata
        }
        
        # Use broadcast_object_list for complex data
        broadcast_data = [data]
        dist.broadcast_object_list(broadcast_data, src=0)
        
        logger.info(f"[Rank 0] Broadcast work: {WorkType(signal.work_type).name}")
    
    def receive_work(self) -> WorkSignal:
        """Receive work signal (called by non-zero ranks)."""
        if self.rank == 0:
            raise RuntimeError("Rank 0 should not receive work, it broadcasts")
        
        # Receive work type
        work_type_tensor = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.broadcast(work_type_tensor, src=0)
        work_type = int(work_type_tensor.item())
        
        # Receive data
        broadcast_data = [None]
        dist.broadcast_object_list(broadcast_data, src=0)
        data = broadcast_data[0]
        
        signal = WorkSignal(
            work_type=work_type,
            request_id=data.get('request_id', ''),
            context=data.get('context', ''),
            appended_text=data.get('appended_text', ''),
            prompt=data.get('prompt', ''),  # For simple synthesis
            max_tokens=data.get('max_tokens', 500),
            temperature=data.get('temperature', 0.7),
            metadata=data.get('metadata')
        )
        
        logger.info(f"[Rank {self.rank}] Received work: {WorkType(work_type).name}")
        return signal
    
    def worker_loop(self):
        """Main work loop for non-zero ranks."""
        logger.info(f"[Rank {self.rank}] Entering work loop")
        
        while self.running:
            try:
                # Wait for work signal from rank 0
                signal = self.receive_work()
                
                work_type = WorkType(signal.work_type)
                
                if work_type == WorkType.SHUTDOWN:
                    logger.info(f"[Rank {self.rank}] Received shutdown signal")
                    self.running = False
                    break
                
                elif work_type == WorkType.PREFILL_INITIAL:
                    # Execute prefill
                    self._prefill_initial_impl(
                        signal.request_id,
                        signal.context,
                        signal.metadata
                    )
                
                elif work_type == WorkType.SYNTHESIZE_FINAL:
                    # Execute synthesis
                    self._synthesize_final_impl(
                        signal.request_id,
                        signal.appended_text,
                        signal.max_tokens,
                        signal.temperature,
                        signal.metadata
                    )
                
                elif work_type == WorkType.SIMPLE_SYNTHESIS:
                    # Execute simple synthesis
                    self._simple_synthesis_impl(
                        signal.request_id,
                        signal.prompt,
                        signal.max_tokens,
                        signal.temperature,
                        signal.metadata
                    )
                
            except Exception as e:
                logger.error(f"[Rank {self.rank}] Error in work loop: {e}", exc_info=True)
                # Continue running even if one request fails
        
        logger.info(f"[Rank {self.rank}] Exiting work loop")
    
    # ========================================================================
    # Phase 1: Initial Prefill
    # ========================================================================
    
    async def prefill_initial(
        self,
        request_id: str,
        context: str,
        metadata: Optional[Dict] = None
    ) -> InitialPrefillResponse:
        """
        Phase 1: Prefill initial context and store KV cache.
        
        This is called by rank 0 when HTTP request arrives.
        It broadcasts work to all ranks and coordinates execution.
        """
        if self.rank != 0:
            raise RuntimeError("Only rank 0 should handle HTTP requests")
        
        # Check for session conflict
        if self.active_session_id is not None:
            raise ValueError(
                f"Session conflict: {self.active_session_id} already active. "
                f"Max concurrent sessions: 1"
            )
        self.active_session_id = request_id
        
        # Broadcast work to all ranks
        signal = WorkSignal(
            work_type=WorkType.PREFILL_INITIAL.value,
            request_id=request_id,
            context=context,
            metadata=metadata
        )
        self.broadcast_work(signal)
        
        # Execute on rank 0
        response = self._prefill_initial_impl(request_id, context, metadata)
        
        return response
    
    def _prefill_initial_impl(
        self,
        request_id: str,
        context: str,
        metadata: Optional[Dict] = None
    ) -> Optional[InitialPrefillResponse]:
        """
        Implementation of prefill that runs on all ranks.
        Only rank 0 returns a response.
        """
        start_time = time.time()
        gpu_id = f"a100_gpu{self.rank}"
        
        with ProfilerContext(
            gpu_id=gpu_id,
            request_id=request_id,
            enabled=self.profiling_enabled
        ):
            # Tokenize context (all ranks do this)
            with profile_operation("tokenize", gpu_id, request_id, self.profiling_enabled):
                encoded = self.tokenizer(
                    context,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
            
            seq_len = input_ids.shape[1]
            
            # Prefill with TP (all ranks participate)
            with profile_operation("prefill", gpu_id, request_id, self.profiling_enabled):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        return_dict=True
                    )
                    
                    # All-reduce logits
                    if dist.is_initialized():
                        all_reduce_logits(outputs.logits, self.world_size)
                    
                    # Convert DynamicCache to legacy tuple format for Phase 2 compatibility
                    kv_cache = outputs.past_key_values
                    logger.info(f"[Rank {self.rank}] KV cache type before conversion: {type(kv_cache)}")
                    
                    # Check if it's a DynamicCache and convert to tuple
                    if hasattr(kv_cache, '__class__') and 'DynamicCache' in str(type(kv_cache)):
                        logger.info(f"[Rank {self.rank}] Converting DynamicCache to tuple format")
                        # DynamicCache is iterable - each element is already a (k, v) tuple
                        kv_cache = tuple(kv_cache)
                        logger.info(f"[Rank {self.rank}] Converted to tuple with {len(kv_cache)} layers")
            
            # Validate KV cache (skip validation for DynamicCache - it's fine)
            if not (hasattr(kv_cache, '__class__') and 'Cache' in kv_cache.__class__.__name__):
                if not validate_kv_cache(kv_cache):
                    raise ValueError("Invalid KV cache generated")
            
            # Estimate cache size
            cache_size_mb = estimate_kv_cache_size(kv_cache, dtype=self.model.dtype) * 1024  # GB to MB
            
            # Store KV cache (all ranks store locally)
            self.kv_cache_store[request_id] = CacheEntry(
                kv_cache=kv_cache,
                timestamp=time.time(),
                seq_len=seq_len,
                context_preview=context[:100],
                metadata=metadata or {}
            )
            
            logger.info(
                f"[Rank {self.rank}] Stored KV cache for {request_id}: "
                f"{seq_len} tokens, {cache_size_mb:.2f} MB"
            )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Only rank 0 returns response
        if self.rank == 0:
            return InitialPrefillResponse(
                status="success",
                request_id=request_id,
                cache_stored=True,
                context_tokens=seq_len,
                prefill_time_ms=elapsed_ms,
                cache_size_mb=cache_size_mb
            )
        
        return None
    
    # ========================================================================
    # Phase 2: Final Synthesis
    # ========================================================================
    
    async def synthesize_final(
        self,
        request_id: str,
        appended_text: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        metadata: Optional[Dict] = None
    ) -> FinalSynthesisResponse:
        """
        Phase 2: Retrieve KV cache, append new text, and generate synthesis.
        
        This is called by rank 0 when HTTP request arrives.
        It broadcasts work to all ranks and coordinates execution.
        """
        if self.rank != 0:
            raise RuntimeError("Only rank 0 should handle HTTP requests")
        
        # Broadcast work to all ranks
        signal = WorkSignal(
            work_type=WorkType.SYNTHESIZE_FINAL.value,
            request_id=request_id,
            appended_text=appended_text,
            max_tokens=max_tokens,
            temperature=temperature,
            metadata=metadata
        )
        self.broadcast_work(signal)
        
        # Execute on rank 0
        response = self._synthesize_final_impl(
            request_id, appended_text, max_tokens, temperature, metadata
        )
        
        return response
    
    def _synthesize_final_impl(
        self,
        request_id: str,
        appended_text: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        metadata: Optional[Dict] = None
    ) -> Optional[FinalSynthesisResponse]:
        """
        Implementation of synthesis that runs on all ranks.
        Only rank 0 returns a response.
        """
        start_time = time.time()
        gpu_id = f"a100_gpu{self.rank}"
        
        # Retrieve KV cache (all ranks have it locally)
        if request_id not in self.kv_cache_store:
            raise ValueError(f"No KV cache found for request_id: {request_id}")
        
        cache_entry = self.kv_cache_store[request_id]
        kv_cache = cache_entry.kv_cache
        
        # Check if expired
        age = time.time() - cache_entry.timestamp
        if age > self.cache_timeout:
            del self.kv_cache_store[request_id]
            if self.rank == 0:
                self.active_session_id = None
            raise ValueError(
                f"KV cache expired for {request_id} "
                f"(age: {age:.1f}s, timeout: {self.cache_timeout}s)"
            )
        
        logger.info(f"[Rank {self.rank}] Retrieved KV cache for {request_id} (age: {age:.1f}s)")
        
        with ProfilerContext(
            gpu_id=gpu_id,
            request_id=request_id + "_synthesis",
            enabled=self.profiling_enabled
        ):
            # Tokenize appended text
            with profile_operation("tokenize_append", gpu_id, request_id, self.profiling_enabled):
                encoded = self.tokenizer(
                    appended_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    add_special_tokens=False  # Appending to existing sequence
                )
                new_input_ids = encoded["input_ids"].to(self.device)
            
            prefill_start = time.time()
            
            # Incremental prefill on new tokens
            with profile_operation("incremental_prefill", gpu_id, request_id, self.profiling_enabled):
                with torch.no_grad():
                    # Get past sequence length from tuple format
                    past_length = kv_cache[0][0].shape[2]  # Shape: [batch, heads, seq_len, head_dim]
                    new_length = new_input_ids.shape[1]
                    
                    # Create proper attention mask: full length (past + new)
                    total_length = past_length + new_length
                    attention_mask = torch.ones(
                        (1, total_length),
                        dtype=torch.long,
                        device=self.device
                    )
                    
                    outputs = self.model(
                        input_ids=new_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=kv_cache,  # Tuple format!
                        use_cache=True,
                        return_dict=True
                    )
                    
                    # All-reduce logits
                    if dist.is_initialized():
                        all_reduce_logits(outputs.logits, self.world_size)
                    
                    # KV cache is already in tuple format (from Phase 1 conversion)
                    extended_kv_cache = outputs.past_key_values
            
            prefill_time_ms = (time.time() - prefill_start) * 1000
            decode_start = time.time()
            
            # Generate synthesis
            with profile_operation("decode_synthesis", gpu_id, request_id, self.profiling_enabled):
                generated_tokens = []
                current_kv = extended_kv_cache
                
                # Start from last token logits
                logits = outputs.logits[:, -1, :]
                
                for step in range(max_tokens):
                    # Sample next token
                    if temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    token_id = next_token.item()
                    generated_tokens.append(token_id)
                    
                    # Check for EOS
                    if token_id == self.tokenizer.eos_token_id:
                        break
                    
                    # Forward pass for next token
                    with torch.no_grad():
                        # Get current sequence length from tuple cache
                        current_length = current_kv[0][0].shape[2] + 1  # Adding 1 for new token
                        
                        # Attention mask for full sequence
                        attention_mask = torch.ones(
                            (1, current_length),
                            dtype=torch.long,
                            device=self.device
                        )
                        
                        outputs = self.model(
                            input_ids=next_token.unsqueeze(0),
                            attention_mask=attention_mask,
                            past_key_values=current_kv,
                            use_cache=True,
                            return_dict=True
                        )
                        
                        # All-reduce logits
                        if dist.is_initialized():
                            all_reduce_logits(outputs.logits, self.world_size)
                        
                        # Keep KV cache as-is
                        current_kv = outputs.past_key_values
                        
                        logits = outputs.logits[:, -1, :]
                
                # Decode generated text
                synthesis_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            
            decode_time_ms = (time.time() - decode_start) * 1000
        
        # Cleanup (all ranks)
        if request_id in self.kv_cache_store:
            del self.kv_cache_store[request_id]
            logger.info(f"[Rank {self.rank}] Cleaned up KV cache for {request_id}")
        
        if self.rank == 0:
            self.active_session_id = None
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Only rank 0 returns response
        if self.rank == 0:
            return FinalSynthesisResponse(
                status="success",
                request_id=request_id,
                synthesis=synthesis_text,
                num_tokens=len(generated_tokens),
                prefill_time_ms=prefill_time_ms,
                decode_time_ms=decode_time_ms,
                total_time_ms=total_time_ms
            )
        
        return None
    
    # ========================================================================
    # Simple Synthesis (No KV Cache Reuse)
    # ========================================================================
    
    async def simple_synthesis(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        metadata: Optional[Dict] = None
    ) -> Optional[SimpleSynthesisResponse]:
        """
        Simple single-phase synthesis: prefill + decode with no KV cache reuse.
        
        This is simpler and avoids DynamicCache complexity.
        Just takes a full prompt and generates text.
        """
        if self.rank == 0:
            # Broadcast work to all ranks
            signal = WorkSignal(
                work_type=WorkType.SIMPLE_SYNTHESIS.value,
                request_id=request_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                metadata=metadata
            )
            self.broadcast_work(signal)
        
        # All ranks process
        response = self._simple_synthesis_impl(request_id, prompt, max_tokens, temperature, metadata)
        
        return response
    
    def _simple_synthesis_impl(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        metadata: Optional[Dict]
    ) -> Optional[SimpleSynthesisResponse]:
        """Implementation of simple synthesis."""
        start_time = time.time()
        gpu_id = f"a100_gpu{self.rank}"
        
        with ProfilerContext(
            gpu_id=gpu_id,
            request_id=request_id,
            enabled=self.profiling_enabled
        ):
            # Tokenize
            with profile_operation("tokenize", gpu_id, request_id, self.profiling_enabled):
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
            
            prefill_start = time.time()
            
            # Prefill
            with profile_operation("prefill", gpu_id, request_id, self.profiling_enabled):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,  # Don't use KV cache to avoid dimension issues
                        return_dict=True
                    )
                    logits = outputs.logits[:, -1, :]
                    
                    # Synchronize logits across all ranks
                    if dist.is_initialized():
                        all_reduce_logits(logits, self.world_size)
            
            prefill_time_ms = (time.time() - prefill_start) * 1000
            decode_start = time.time()
            
            # Decode without KV cache (simpler, avoids all dimension issues)
            with profile_operation("decode", gpu_id, request_id, self.profiling_enabled):
                generated_tokens = []
                current_ids = input_ids.clone()
                
                for step in range(max_tokens):
                    # Sample next token from synchronized logits
                    if temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    token_id = next_token.item()
                    generated_tokens.append(token_id)
                    
                    # Check for EOS
                    if token_id == self.tokenizer.eos_token_id:
                        break
                    
                    # Append token and forward pass
                    with torch.no_grad():
                        current_ids = torch.cat([current_ids, next_token], dim=1)
                        
                        outputs = self.model(
                            input_ids=current_ids,
                            use_cache=False,
                            return_dict=True
                        )
                        logits = outputs.logits[:, -1, :]
                        
                        # Synchronize logits across all ranks
                        if dist.is_initialized():
                            all_reduce_logits(logits, self.world_size)
                
                # Decode generated text
                generated_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            
            decode_time_ms = (time.time() - decode_start) * 1000
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Only rank 0 returns response
        if self.rank == 0:
            logger.info(
                f"[Rank {self.rank}] Simple synthesis complete: "
                f"{len(generated_tokens)} tokens in {total_time_ms:.2f}ms"
            )
            return SimpleSynthesisResponse(
                status="success",
                request_id=request_id,
                text=generated_text,
                num_tokens=len(generated_tokens),
                prefill_time_ms=prefill_time_ms,
                decode_time_ms=decode_time_ms,
                total_time_ms=total_time_ms
            )
        
        return None
    
    # ========================================================================
    # Cache Management
    # ========================================================================
    
    async def _cache_cleanup_task(self):
        """Background task to clean up expired KV caches (rank 0 only)."""
        while self.running:
            await asyncio.sleep(60)  # Check every minute
            
            current_time = time.time()
            expired_ids = []
            
            for req_id, cache_entry in self.kv_cache_store.items():
                age = current_time - cache_entry.timestamp
                if age > self.cache_timeout:
                    expired_ids.append(req_id)
            
            for req_id in expired_ids:
                del self.kv_cache_store[req_id]
                logger.warning(
                    f"[Rank {self.rank}] Expired KV cache for {req_id} "
                    f"after {self.cache_timeout}s timeout"
                )
                
                if self.active_session_id == req_id:
                    self.active_session_id = None
    
    async def get_cache_status(self) -> Dict:
        """Get current cache status (debug endpoint, rank 0 only)."""
        current_time = time.time()
        caches = []
        
        for req_id, cache_entry in self.kv_cache_store.items():
            age = current_time - cache_entry.timestamp
            caches.append({
                "request_id": req_id,
                "age_seconds": age,
                "seq_len": cache_entry.seq_len,
                "context_preview": cache_entry.context_preview,
                "expires_in": self.cache_timeout - age
            })
        
        return {
            "active_caches": caches,
            "active_session_id": self.active_session_id,
            "cache_timeout": self.cache_timeout
        }
    
    def shutdown(self):
        """Shutdown worker (rank 0 broadcasts shutdown signal)."""
        self.running = False
        
        if self.rank == 0:
            signal = WorkSignal(work_type=WorkType.SHUTDOWN.value)
            self.broadcast_work(signal)
            logger.info("[Rank 0] Broadcast shutdown signal")


# ============================================================================
# HTTP Server (Rank 0 Only)
# ============================================================================

def create_synthesis_app(worker: SynthesisWorker):
    """Create FastAPI app with synthesis endpoints."""
    app = FastAPI(
        title="A100 Synthesis Server",
        description="4-way Tensor Parallel Synthesis with KV Cache Reuse",
        version="1.0.0"
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "rank": worker.rank,
            "world_size": worker.world_size,
            "model": worker.model_name,
            "active_session": worker.active_session_id
        }
    
    @app.post("/prefill_initial", response_model=InitialPrefillResponse)
    async def prefill_initial(request: InitialPrefillRequest):
        """Phase 1: Prefill initial context."""
        try:
            response = await worker.prefill_initial(
                request_id=request.request_id,
                context=request.context,
                metadata=request.metadata
            )
            return response
        except Exception as e:
            logger.error(f"Error in prefill_initial: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/synthesize_final", response_model=FinalSynthesisResponse)
    async def synthesize_final(request: FinalSynthesisRequest):
        """Phase 2: Synthesize final answer with KV cache reuse."""
        try:
            response = await worker.synthesize_final(
                request_id=request.request_id,
                appended_text=request.appended_text,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                metadata=request.metadata
            )
            return response
        except Exception as e:
            logger.error(f"Error in synthesize_final: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/synthesize", response_model=SimpleSynthesisResponse)
    async def simple_synthesize(request: SimpleSynthesisRequest):
        """Simple synthesis: prefill + decode in one go (no KV cache reuse)."""
        try:
            response = await worker.simple_synthesis(
                request_id=request.request_id,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                metadata=request.metadata
            )
            return response
        except Exception as e:
            logger.error(f"Error in simple_synthesis: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/debug/cache_status")
    async def cache_status():
        """Debug: Get current cache status."""
        return await worker.get_cache_status()
    
    @app.get("/debug/tp_status")
    async def tp_status():
        """Debug: Get tensor parallel status."""
        tp_info = get_tp_group_info()
        return {
            "rank": worker.rank,
            "world_size": worker.world_size,
            "device": worker.device,
            "model": worker.model_name,
            "tp_info": tp_info
        }
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on server shutdown."""
        worker.shutdown()
    
    return app


# ============================================================================
# Worker Process
# ============================================================================

def worker_process(rank: int, world_size: int, args):
    """Worker process for one GPU."""
    # Initialize distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '29500')
    os.environ['NCCL_DEBUG'] = os.getenv('NCCL_DEBUG', 'WARN')
    
    dist.init_process_group(
        backend='nccl',
        init_method=f"tcp://localhost:{os.environ['MASTER_PORT']}",
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    
    # Create worker
    worker = SynthesisWorker(
        rank=rank,
        world_size=world_size,
        model_name=args.model_name,
        precision=args.precision,
        port=args.port,
        profiling_enabled=args.profiling_enabled,
        cache_timeout=args.cache_timeout
    )
    
    # Initialize model
    worker.initialize()
    
    # Only rank 0 runs HTTP server
    if rank == 0:
        app = create_synthesis_app(worker)
        
        # Start cache cleanup task
        @app.on_event("startup")
        async def startup_event():
            asyncio.create_task(worker._cache_cleanup_task())
        
        logger.info(f"[Rank 0] Starting HTTP server on port {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        # Other ranks enter work loop
        logger.info(f"[Rank {rank}] Entering worker loop")
        worker.worker_loop()
        logger.info(f"[Rank {rank}] Worker loop exited")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="A100 Synthesis Server with 4-way TP")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-70b-chat-hf")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--profiling-enabled", action="store_true")
    parser.add_argument("--cache-timeout", type=int, default=300, help="KV cache timeout in seconds")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("A100 Synthesis Server - 4-way Tensor Parallelism with NCCL Broadcast")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Precision: {args.precision}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Cache timeout: {args.cache_timeout}s")
    logger.info(f"Profiling: {args.profiling_enabled}")
    logger.info("=" * 70)
    
    # Launch 4 worker processes
    world_size = 4
    
    mp.spawn(
        worker_process,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
