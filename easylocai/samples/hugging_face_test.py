from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":

    MODEL = "openai/gpt-oss-20b"  # HF repo id

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Requires `accelerate` when using device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",  # shards across available devices
        dtype="auto",  # replaces torch_dtype
    )

    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
