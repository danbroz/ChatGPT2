#!/usr/bin/env python3

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, AdamW
from rlhf.reward_model import GPT2RewardModel

class RewardPairwiseDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        resp1 = item["response_1"]
        resp2 = item["response_2"]
        chosen = item["chosen"]  # 1 or 2

        # Tokenize (prompt + response1) and (prompt + response2)
        # We'll feed them separately to the reward model
        tokens_1 = self.tokenizer.encode(prompt + "\nAssistant: " + resp1, truncation=True, max_length=self.max_length)
        tokens_2 = self.tokenizer.encode(prompt + "\nAssistant: " + resp2, truncation=True, max_length=self.max_length)

        return {
            "input_ids_1": torch.tensor(tokens_1, dtype=torch.long),
            "input_ids_2": torch.tensor(tokens_2, dtype=torch.long),
            "chosen": chosen
        }

def collate_reward(batch):
    # We need to pad each input_ids_1, input_ids_2 to the max length in the batch
    max_len_1 = max(x["input_ids_1"].size(0) for x in batch)
    max_len_2 = max(x["input_ids_2"].size(0) for x in batch)

    input_ids_1 = torch.zeros((len(batch), max_len_1), dtype=torch.long)
    input_ids_2 = torch.zeros((len(batch), max_len_2), dtype=torch.long)
    attention_mask_1 = torch.zeros((len(batch), max_len_1), dtype=torch.long)
    attention_mask_2 = torch.zeros((len(batch), max_len_2), dtype=torch.long)
    chosen_list = []

    for i, example in enumerate(batch):
        len1 = example["input_ids_1"].size(0)
        len2 = example["input_ids_2"].size(0)
        input_ids_1[i, :len1] = example["input_ids_1"]
        input_ids_2[i, :len2] = example["input_ids_2"]
        attention_mask_1[i, :len1] = 1
        attention_mask_2[i, :len2] = 1
        chosen_list.append(example["chosen"])

    return {
        "input_ids_1": input_ids_1,
        "input_ids_2": input_ids_2,
        "attention_mask_1": attention_mask_1,
        "attention_mask_2": attention_mask_2,
        "chosen": torch.tensor(chosen_list, dtype=torch.long)
    }

def main():
    reward_model_save = "./models/reward_model"
    base_model = "gpt2"  # or your supervised model path
    data_file = "data/reward_data.json"
    batch_size = 2
    lr = 1e-5
    epochs = 1

    tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    model = GPT2RewardModel(base_model=base_model)

    dataset = RewardPairwiseDataset(tokenizer, data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_reward)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            input_ids_1 = batch["input_ids_1"].to(device)
            input_ids_2 = batch["input_ids_2"].to(device)
            attn_1 = batch["attention_mask_1"].to(device)
            attn_2 = batch["attention_mask_2"].to(device)
            chosen = batch["chosen"].to(device)

            # Forward pass
            rewards_1 = model(input_ids_1, attention_mask=attn_1)
            rewards_2 = model(input_ids_2, attention_mask=attn_2)

            # We want the chosen response to have a higher reward
            # simple hinge or margin-based approach
            # L = -log( sigmoid( reward_chosen - reward_not_chosen ) ) is typical
            # We'll do something simple:
            chosen_mask = (chosen == 1).float()
            not_chosen_mask = (chosen == 2).float()
            reward_chosen = rewards_1 * chosen_mask + rewards_2 * not_chosen_mask
            reward_not_chosen = rewards_2 * chosen_mask + rewards_1 * not_chosen_mask

            # Bradley-Terry style pairwise loss
            loss = -torch.log(torch.sigmoid(reward_chosen - reward_not_chosen)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")

    os.makedirs(reward_model_save, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(reward_model_save, "reward_model.pt"))
    tokenizer.save_pretrained(reward_model_save)
    print(f"Reward model saved to {reward_model_save}")

if __name__ == "__main__":
    main()
