#!/usr/bin/env python3

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from rlhf.reward_model import GPT2RewardModel
from rlhf.rlhf_trainer import PPOTrainer
from copy import deepcopy

def main():
    # Paths
    supervised_model_path = "./models/supervised_model"
    reward_model_path = "./models/reward_model"
    rlhf_model_save_path = "./models/rlhf_policy"
    
    # Hyperparams
    lr = 1e-6
    kl_coef = 0.1
    ppo_epochs = 1
    batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy (supervised fine-tuned GPT-2)
    policy = GPT2LMHeadModel.from_pretrained(supervised_model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(supervised_model_path)

    # Load reference policy (frozen)
    ref_policy = deepcopy(policy).eval()
    for param in ref_policy.parameters():
        param.requires_grad = False

    # Load reward model
    reward_model = GPT2RewardModel()
    reward_model.load_state_dict(torch.load(os.path.join(reward_model_path, "reward_model.pt"), map_location=device))
    reward_model.to(device)
    reward_model.eval()

    policy.to(device)
    
    policy_optimizer = AdamW(policy.parameters(), lr=lr)

    # Create PPO trainer
    ppo_trainer = PPOTrainer(policy, ref_policy, reward_model, policy_optimizer, tokenizer,
                             kl_coef=kl_coef, ppo_epochs=ppo_epochs, mini_batch_size=batch_size)

    # We'll read some "prompt" lines from a file, then do PPO updates
    # In a real project, you'd sample user prompts from a dataset or environment
    prompts_file = "data/prompts_for_rlhf.txt"
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Let's do a single pass over the prompts in small batches
    # Real training would iterate multiple times over more data
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        train_info = ppo_trainer.train_step(batch_prompts)
        print(f"Train Step: {i}, Avg Reward: {train_info['avg_reward']:.4f}")

    # Save the final PPO-tuned policy
    os.makedirs(rlhf_model_save_path, exist_ok=True)
    policy.save_pretrained(rlhf_model_save_path)
    tokenizer.save_pretrained(rlhf_model_save_path)
    print(f"RLHF-tuned policy saved to {rlhf_model_save_path}")

if __name__ == "__main__":
    main()
