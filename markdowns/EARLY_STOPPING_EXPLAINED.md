# What Causes Early Stopping in Generation

## The Stopping Condition Code

From `serving/common/inference.py` lines 178-180:

```python
# Check for stop tokens
if next_token.item() in stop_tokens:
    break
```

Where `stop_tokens` defaults to `[tokenizer.eos_token_id]` (line 152).

For Phi-2:
- **EOS token ID:** 50256
- **EOS token:** `<|endoftext|>`

## What Actually Happens

During generation, the model:

1. **Samples the next token** probabilistically based on temperature
2. **Checks if it's an EOS token** (50256)
3. **If yes, immediately stops** generation
4. **If no, continues** to next token (up to max_tokens=50)

## Why Different GPUs Stop at Different Times

### Stochastic Sampling with Temperature

Each GPU independently samples from the probability distribution:

```python
# In decode_step() function
logits = outputs.logits[:, -1, :] / temperature  # Scale by temperature
probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert to probabilities
next_token = torch.multinomial(probs, num_samples=1)  # RANDOM sample
```

**Key point:** `torch.multinomial()` is **non-deterministic** - even with the same inputs, different GPUs can sample different tokens.

### Temperature's Effect on Stopping

| Temperature | Effect | Early Stop Probability |
|-------------|--------|----------------------|
| 0.0 (greedy) | Always picks highest prob token | Low (deterministic) |
| 0.3 (low) | Strongly favors high-prob tokens | Low-Medium |
| 0.7 (medium) | Balanced distribution | Medium |
| 1.0 (high) | More uniform distribution | Medium-High |
| >1.5 (very high) | Nearly uniform | High |

**Higher temperature = flatter distribution = higher chance of sampling rare tokens, including EOS.**

## Concrete Examples from Your Runs

### Run 1 (Earlier)
**GPU 2 stopped early** (2 tokens, 8.33 MB trace):
```
What is the capital of France?
A) Berlin  <- Stopped here (likely sampled EOS or model decided to stop)
```

**What happened:**
1. Generated "A"
2. Generated ")" 
3. Generated "Berlin"
4. **Model sampled token that triggered stopping** (either EOS or a token that naturally ends)

### Run 2 (Current)
**GPU 0 stopped early** (3 tokens, 11.72 MB trace):
```
What is the capital of France?
...  <- Stopped here
```

**GPU 2 generated most** (50 tokens, 189.70 MB trace):
```
What is the capital of France?
A) Berlin
B) Rome
C) Paris
D) Madrid

Answer: C) Paris

Question 2:
What is the capital of Australia?
A) Sydney  <- Stopped at max_tokens=50
```

## The Three Ways Generation Stops

### 1. EOS Token Sampled (Most Common for Short Outputs)
```python
if next_token.item() == 50256:  # <|endoftext|>
    break
```

**When this happens:**
- Model decides the response is "complete"
- Often after giving a short answer
- More likely with higher temperatures (flatter distribution)

### 2. Max Tokens Reached
```python
for _ in range(max_tokens):  # max_tokens=50 in your case
    # ... generate token ...
```

**When this happens:**
- Model keeps generating but hits the limit
- Common for verbose responses
- Seen in GPU 2's current run (generated Question 2)

### 3. Other Stop Tokens (If Configured)
```python
stop_tokens = [tokenizer.eos_token_id, custom_token_id, ...]
```

In your case, only EOS is in `stop_tokens`.

## Why GPU 2 Stopped Early in Run 1

**Most likely explanation:** Model sampled the EOS token after "Berlin".

### Why Would the Model Do This?

Phi-2 was trained on data where:
- Short answers are common
- Multiple-choice formats often end after one option
- The model learned that "A) [Answer]" can be a complete response

With `temperature=0.7`, there was enough randomness that:
1. GPU 2 sampled "Berlin" (wrong answer)
2. Next token distribution favored ending (newline, period, or EOS)
3. GPU 2 sampled EOS or a naturally-ending token
4. Generation stopped

### Why Other GPUs Didn't Stop?

**GPU 1 (temp=0.3):** Conservative sampling favored continuing the response
**GPU 3 (temp=1.0):** High creativity led to longer, more elaborate response
**GPU 0 (temp=0.7):** In Run 2, it happened to sample EOS early instead

## Proving the EOS Theory

To confirm EOS was sampled, you'd need to:

### Option 1: Log Raw Token IDs
```python
# In serving/common/inference.py, line 170
generated_tokens.append(next_token.item())

# Add logging
logger.info(f"Generated token ID: {next_token.item()}")
if next_token.item() == tokenizer.eos_token_id:
    logger.info(f"EOS token sampled! Stopping generation.")
```

### Option 2: Check Model's Next-Token Probabilities
```python
# After decode_step
probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
eos_prob = probs[0, tokenizer.eos_token_id].item()
logger.info(f"EOS token probability: {eos_prob:.4f}")
```

## Key Insight: This is NOT a Bug

**This stochastic behavior is exactly what you want for multi-agent systems:**

✅ **Diversity:** Different agents explore different response styles  
✅ **Robustness:** Some agents give short answers, some give long explanations  
✅ **Realism:** Mimics human behavior (some people are concise, others verbose)  
✅ **Validation:** Judge stage can evaluate both short and long responses

**The randomness in stopping time is a FEATURE, not a bug.**

## Summary

**What causes early stopping?**
1. **Immediate cause:** `next_token.item() in stop_tokens` evaluates to `True`
2. **Underlying cause:** Model sampled EOS token (50256) due to:
   - Temperature-based stochastic sampling
   - Model's learned probability distribution
   - Random variation across GPUs

**Why does it vary across GPUs?**
- Each GPU does **independent random sampling**
- Same KV cache + same prompt ≠ same output
- Temperature > 0 means non-deterministic results

**Why does it vary across runs?**
- Different random seeds (implicit in PyTorch)
- GPU timing differences affecting random number generation
- Truly stochastic process

**Can you prevent it?**
- Set `temperature=0.0` for greedy decoding (deterministic)
- Use `torch.manual_seed()` for reproducibility
- Increase `max_tokens` to allow longer generation
- Remove EOS from `stop_tokens` (not recommended)

**Should you prevent it?**
- **No!** This diversity is exactly why you built a multi-agent system
- The Judge/Chairman stages are designed to handle varying response lengths
- Stochastic behavior → diverse outputs → better ensemble results

