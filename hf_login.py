#!/usr/bin/env python3
"""
HuggingFace login helper script.
"""

from huggingface_hub import login
import sys

print("="*70)
print("HuggingFace Login")
print("="*70)
print()
print("Instructions:")
print("1. Go to: https://huggingface.co/settings/tokens")
print("2. Click 'New token'")
print("3. Name: 'llama-inference', Type: 'Read'")
print("4. Copy the token (starts with 'hf_...')")
print()

token = input("Paste your HuggingFace token here: ").strip()

if not token:
    print("✗ No token provided")
    sys.exit(1)

if not token.startswith("hf_"):
    print("⚠ Warning: Token should start with 'hf_'")
    print("Are you sure this is correct?")
    confirm = input("Continue anyway? (y/n): ").strip().lower()
    if confirm != 'y':
        sys.exit(1)

try:
    print("\nLogging in...")
    login(token=token, add_to_git_credential=False)
    print("✓ Login successful!")
    print(f"✓ Token saved to: ~/.cache/huggingface/token")
    print()
    
    # Test access to Llama-2
    print("Testing access to Llama-2-70B...")
    from transformers import AutoConfig
    
    try:
        config = AutoConfig.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
        print("✓ Success! You have access to Llama-2-70b-chat-hf")
        print(f"✓ Model size: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
        print()
        print("="*70)
        print("You're all set! You can now use Llama-2-70B.")
        print("="*70)
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print("⚠ Token works, but you need to request access to Llama-2")
            print()
            print("Steps:")
            print("1. Visit: https://huggingface.co/meta-llama/Llama-2-70b-chat-hf")
            print("2. Click 'Agree and access repository'")
            print("3. Wait for approval (usually instant)")
            print("4. Run this script again to verify")
        else:
            print(f"⚠ Error testing access: {e}")
            print("Your token is saved, but there may be an issue.")
    
except Exception as e:
    print(f"✗ Login failed: {e}")
    sys.exit(1)

