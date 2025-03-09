#!/usr/bin/env python3

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW

# Hypothetical dataset class that loads multi-turn chat data in text form
class ChatDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=1024):
        # file_path is a text file (or multiple) that has the multi-turn dialogues
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.read().split("<|begin_of_conversation|>")

        self.examples = []
        for conversation in self.lines:
            conversation = conversation.strip()
            if not conversation:
                continue
            # add special token if you like
            conversation_text = "<|begin_of_conversation|> " + conversation

            # tokenize
            tokens = tokenizer.encode(conversation_text, add_special_tokens=True)
            
            # chunk into block_size
            for i in range(0, len(tokens), block_size):
                chunk = tokens[i:i + block_size]
                self.examples.append(chunk)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

def collate_fn(examples):
    # Collate into a batch of equal-length sequences with padding
    length = max([ex.size(0) for ex in examples])
    input_ids = torch.zeros((len(examples), length), dtype=torch.long)
    attention_mask = torch.zeros((len(examples), length), dtype=torch.long)

    for i, ex in enumerate(examples):
        input_ids[i, :ex.size(0)] = ex
        attention_mask[i, :ex.size(0)] = 1
    return input_ids, attention_mask

def main():
    # Hyperparameters
    train_file = "data/chat_data.txt"  # your multi-turn conversation dataset
    model_name_or_path = "gpt2"
    output_dir = "./trained_model"
    epochs = 2
    batch_size = 2
    lr = 5e-5

    # Load tokenizer, model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    config = GPT2Config.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config)

    # Possibly add special tokens
    special_tokens_dict = {
        'additional_special_tokens': [
            '<|begin_of_conversation|>',
            '<|end_of_conversation|>',
            'User:',
            'Assistant:'
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Dataset, DataLoader
    train_dataset = ChatDataset(tokenizer, train_file)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask = [t.to(device) for t in batch]

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item()}")

    # Save the finetuned model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved to", output_dir)

if __name__ == "__main__":
    main()
