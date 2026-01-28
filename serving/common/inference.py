"""Inference primitives for prefill and decode operations."""

from typing import Optional, Tuple, Dict, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def prefill(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, Any]:
    """
    Perform prefill phase to generate KV cache.
    
    Args:
        model: The language model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        use_cache: Whether to return KV cache
    
    Returns:
        Tuple of (logits, past_key_values/kv_cache)
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
        )
    
    return outputs.logits, outputs.past_key_values


def decode_step(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    past_key_values: Any,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
) -> Tuple[torch.Tensor, Any]:
    """
    Perform a single decode step.
    
    Args:
        model: The language model
        input_ids: Input token IDs (typically last token) [batch_size, 1]
        past_key_values: KV cache from previous steps
        attention_mask: Attention mask
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        Tuple of (next_token_id, updated_past_key_values)
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
    
    # Get logits for last token
    logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
    
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
    
    # Sample next token
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token, outputs.past_key_values


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    past_key_values: Optional[Any] = None,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    stop_tokens: Optional[list] = None,
) -> Tuple[torch.Tensor, list]:
    """
    Generate tokens autoregressively.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_ids: Input token IDs [batch_size, seq_len]
        past_key_values: Optional pre-computed KV cache
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        stop_tokens: List of token IDs that stop generation
    
    Returns:
        Tuple of (generated_token_ids, generated_tokens_list)
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    # Initialize with input
    generated_ids = input_ids.clone()
    
    # Create attention mask
    attention_mask = torch.ones_like(generated_ids)
    
    # If no KV cache provided, do prefill
    if past_key_values is None:
        _, past_key_values = prefill(model, input_ids, attention_mask)
        # Start generating from next position
        next_input_ids = input_ids[:, -1:]
    else:
        # Use provided KV cache
        next_input_ids = input_ids[:, -1:]
    
    # Stop tokens
    if stop_tokens is None:
        stop_tokens = [tokenizer.eos_token_id]
    
    # Generation loop
    generated_tokens = []
    for _ in range(max_tokens):
        # Decode one step
        next_token, past_key_values = decode_step(
            model=model,
            input_ids=next_input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        
        # Append to generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        generated_tokens.append(next_token.item())
        
        # Update attention mask
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        ], dim=1)
        
        # Check for stop tokens
        if next_token.item() in stop_tokens:
            break
        
        # Update input for next iteration
        next_input_ids = next_token
    
    return generated_ids, generated_tokens


def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
) -> list[str]:
    """
    Generate completions for a batch of prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompt strings
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        List of generated text strings
    """
    device = next(model.parameters()).device
    
    # Tokenize prompts
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    # Generate
    generated_ids, _ = generate(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    
    # Decode
    generated_texts = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    
    return generated_texts


def format_prompt_for_model(
    prompt: str,
    model_name: str,
    system_message: Optional[str] = None
) -> str:
    """
    Format prompt according to model's chat template.
    
    Args:
        prompt: User prompt
        model_name: Model name to determine format
        system_message: Optional system message
    
    Returns:
        Formatted prompt string
    """
    # Check model type and format accordingly
    if "llama" in model_name.lower() or "llama-2" in model_name.lower():
        # Llama 2 chat format
        if system_message:
            return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            return f"<s>[INST] {prompt} [/INST]"
    
    elif "mistral" in model_name.lower():
        # Mistral Instruct format
        if system_message:
            return f"<s>[INST] {system_message}\n\n{prompt} [/INST]"
        else:
            return f"<s>[INST] {prompt} [/INST]"
    
    else:
        # Generic format
        if system_message:
            return f"{system_message}\n\n{prompt}"
        else:
            return prompt

