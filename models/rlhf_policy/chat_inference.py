#!/usr/bin/env python3

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_response(model, tokenizer, conversation_history, max_length=512):
    # We assume conversation_history includes lines with "User:" and "Assistant:"
    # We'll add "Assistant:" before generating the next part
    input_text = conversation_history + "\nAssistant:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.2
        )

    # The generated text includes the prompt plus new tokens
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # extract only the newly generated portion
    return generated_text[len(input_text):].strip()

def main():
    model_path = "./models/rlhf_policy"  # The RLHF-tuned model
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    conversation_history = "System: You are a helpful AI assistant.\n"
    print("Welcome to your RLHF-tuned ChatGPT2. Type 'exit' to quit.")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        conversation_history += f"User: {user_input}\n"
        response = generate_response(model, tokenizer, conversation_history)
        conversation_history += f"Assistant: {response}\n"
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()
