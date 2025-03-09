import torch
import torch.nn.functional as F
from copy import deepcopy

class PPOTrainer:
    """
    A minimal PPO trainer to demonstrate RLHF-like updates.
    """
    def __init__(self, policy, ref_policy, reward_model, policy_optimizer, 
                 tokenizer, kl_coef=0.1, ppo_epochs=1, mini_batch_size=2):
        """
        policy: GPT2LMHeadModel (fine-tuned supervised model)
        ref_policy: a frozen copy of the original supervised model for KL reference
        reward_model: GPT2RewardModel
        tokenizer: ...
        kl_coef: coefficient for KL penalty
        ppo_epochs: how many epochs of PPO updates per batch
        mini_batch_size: ...
        """
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.policy_optimizer = policy_optimizer
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

    def compute_rewards(self, prompts, responses):
        """
        Use reward_model to get reward for each (prompt + response).
        prompts: list of strings
        responses: list of strings
        returns: tensor of shape (batch,)
        """
        device = next(self.reward_model.parameters()).device

        input_texts = [p + "\nAssistant: " + r for p, r in zip(prompts, responses)]
        tokenized = [self.tokenizer.encode(t, return_tensors="pt") for t in input_texts]

        rewards = []
        for toks in tokenized:
            toks = toks.to(device)
            reward = self.reward_model(toks)
            rewards.append(reward.item())  # shape: scalar
        return torch.tensor(rewards, device=device)

    def sample_responses(self, prompts, max_length=100):
        """
        Sample from the current policy to get responses.
        """
        device = next(self.policy.parameters()).device
        responses = []
        logprobs = []
        all_tokens = []

        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt + "\nAssistant:", return_tensors="pt").to(device)
            with torch.no_grad():
                output = self.policy.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                )
            # We'll collect logprobs for the newly generated tokens
            # For simplicity, let's collect them all
            gen_tokens = output[0]
            response_tokens = gen_tokens[len(input_ids[0]):]  # only newly generated part

            # recompute logprobs
            with torch.no_grad():
                out = self.policy(gen_tokens.unsqueeze(0), labels=gen_tokens.unsqueeze(0))
                # out.loss is the cross-entropy for entire sequence
                # but let's do it manually for clarity
                logits = out.logits[:, :-1, :]  # shift for next token
                shift_labels = gen_tokens[1:]
                # gather token logprobs
                token_logprobs = F.log_softmax(logits, dim=-1)
                seq_logprobs = token_logprobs[0, range(logits.size(1)), shift_labels]
                response_logprobs = seq_logprobs[len(input_ids[0]) - 1:]  # only new part
                logprobs.append(response_logprobs.sum().item())

            decoded_response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(decoded_response)
            all_tokens.append(response_tokens.tolist())

        return responses, logprobs, all_tokens

    def compute_kl(self, gen_tokens):
        """
        Compute KL divergence from the reference policy for the newly generated tokens.
        """
        with torch.no_grad():
            ref_out = self.ref_policy(gen_tokens.unsqueeze(0), labels=gen_tokens.unsqueeze(0))
            ref_logits = ref_out.logits[:, :-1, :] 
        out = self.policy(gen_tokens.unsqueeze(0), labels=gen_tokens.unsqueeze(0))
        logits = out.logits[:, :-1, :] 

        # Flatten
        ref_probs = F.log_softmax(ref_logits, dim=-1)
        probs = F.log_softmax(logits, dim=-1)

        kl = (torch.exp(probs) * (probs - ref_probs)).sum(-1).mean()  # average over tokens
        return kl

    def train_step(self, prompts):
        """
        Single PPO update step on a batch of prompts.
        """
        device = next(self.policy.parameters()).device
        # sample from policy
        responses, old_logprobs, token_ids_batch = self.sample_responses(prompts)
        rewards = self.compute_rewards(prompts, responses)

        # For each (prompt, response) do PPO update
        # This is a naive approach: we treat the entire response as a single action
        # so advantage = reward + ...
        # We'll do a single-step advantage estimate for demonstration.
        advantages = rewards - rewards.mean()

        for i in range(len(prompts)):
            # Recompute current logprobs
            gen_tokens = torch.tensor(token_ids_batch[i], device=device)
            # attach prompt tokens
            prompt_ids = self.tokenizer.encode(prompts[i] + "\nAssistant:")
            full_ids = torch.tensor(prompt_ids + token_ids_batch[i], device=device)

            out = self.policy(full_ids.unsqueeze(0), labels=full_ids.unsqueeze(0))
            logits = out.logits[:, :-1, :]
            shift_labels = full_ids[1:]
            token_logprobs = F.log_softmax(logits, dim=-1)
            seq_logprobs = token_logprobs[0, range(logits.size(1)), shift_labels]
            # new logprob is for newly generated
            new_logprob = seq_logprobs[len(prompt_ids) - 1:].sum()

            ratio = torch.exp(new_logprob - old_logprobs[i])
            kl_value = self.compute_kl(full_ids)
            
            # PPO objective
            policy_loss = -ratio * advantages[i]

            # Add KL penalty
            kl_loss = self.kl_coef * kl_value
            total_loss = policy_loss + kl_loss

            self.policy_optimizer.zero_grad()
            total_loss.backward()
            self.policy_optimizer.step()

        return {"avg_reward": rewards.mean().item()}

