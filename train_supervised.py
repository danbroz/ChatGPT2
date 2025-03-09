#!/usr/bin/env python3

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW

class ChatDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=1024):
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = f.read()

        # Split data by a special delimiter or token.
        # For example, assume <|begin_of_conversation|> defines a conversation boundary.
        self.conversations = raw_data.split("<|begin_of_conversation|>")

        self.examples = []
        for conv in self.conversations:
            conv = conv.strip()
            if not conv:
                continue
            # prepend so the model sees the token
            text = "<|begin_of_conversation|> " + conv
            tokens = tokenizer.encode(text, add_special_tokens=True)
            # chunk tokens if needed
            for i in range(0, len(tokens), block_size):
                chunk = tokens[i:i+block_size]
                self.examples.append(chunk)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

def collate_fn(batch):
    max_len = max(x.size(0) for x in batch)
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, example in enumerate(batch):
        length = example.size(0)
        input_ids[i, :length] = example
        attention_mask[i, :length] = 1

    return input_ids, attention_mask

def main():
    train_file = "data/chat_data.txt"
    model_name_or_path = "gpt2"
    output_dir = "./models/supervised_model"
    epochs = 1
    batch_size = 2
    lr = 5e-5

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    config = GPT2Config.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config)

    special_tokens = {
        'additional_special_tokens': [
            '<|begin_of_conversation|>',
            '<|end_of_conversation|>',
            'User:',
            'Assistant:'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    dataset = ChatDataset(tokenizer, train_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            input_ids, attention_mask = [t.to(device) for t in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Supervised model saved to {output_dir}")

if __name__ == "__main__":
    main()
