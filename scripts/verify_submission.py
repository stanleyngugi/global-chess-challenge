from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys

def verify_model(model_path):
    print(f"=== Verifying Submission Artifacts in {model_path} ===")
    
    # 1. Check Files
    required_files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]
    files = os.listdir(model_path)
    for f in required_files:
        if f not in files:
            print(f"❌ MISSING FILE: {f}")
    
    # 2. Check Config (RoPE Scaling)
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        
        # RoPE Check
        if hasattr(config, "rope_scaling"):
            rs = config.rope_scaling
            if rs is None:
                print("⚠️  Warning: rope_scaling is None. Ensure this is intended.")
            elif isinstance(rs, dict):
                rtype = rs.get("type", "unknown")
                if rtype == "llama3":
                    print("❌ FATAL: rope_scaling type is 'llama3'. This WILL CRASH on Trainium vLLM.")
                    print("   ACTION: Patch config.json to use {'type': 'dynamic', 'factor': 8.0}")
                elif rtype == "dynamic":
                    print("✅ RoPE Scaling is 'dynamic' (Trainium Safe).")
                else:
                    print(f"⚠️  Unknown rope_scaling type: {rtype}")
    except Exception as e:
        print(f"❌ Config Load Failed: {e}")

    # 3. Load Model (Smoke Test)
    print("--> Loading model (bf16)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model Load Failed: {e}")
        return

    # 4. NaN Check
    print("--> Scanning for NaNs in weights...")
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ NaN detected in layer: {name}")
            has_nan = True
            break
    if not has_nan:
        print("✅ No NaNs detected.")

    # 5. Tokenizer & Generation Check
    print("--> Testing Tokenizer & Generation...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Check Pad Token
        print(f"   Pad Token ID: {tokenizer.pad_token_id}")
        print(f"   EOS Token ID: {tokenizer.eos_token_id}")
        
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            print("❌ FATAL: pad_token_id == eos_token_id. Model will loop infinitely.")
        elif tokenizer.pad_token_id is None:
             print("❌ FATAL: pad_token_id is None.")
        else:
             print("✅ Pad Token is distinct from EOS.")

        # Generation Test
        prompt = "Position: Start. Best move:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        output_text = tokenizer.decode(outputs[0])
        print(f"   Output: {repr(output_text)}")
        
        if "<|eot_id|>" in output_text or tokenizer.eos_token in output_text:
             print("✅ Model generated stop token.")
        else:
             print("⚠️  Warning: Model did not stop in 20 tokens (expected for short test, but check logs).")

    except Exception as e:
        print(f"❌ Generation Test Failed: {e}")

    print("=== Verification Complete ===")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify_model(sys.argv[1])
    else:
        print("Usage: python verify_submission.py <path_to_merged_model>")
