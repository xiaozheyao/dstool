from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)

chat = [
    {"role": "user", "content": "Hello, how are you?"},
]

output = tokenizer.apply_chat_template(chat, tokenize=False)

print(output)
