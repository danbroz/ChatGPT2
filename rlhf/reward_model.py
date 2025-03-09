import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class GPT2RewardModel(nn.Module):
    """
    A GPT-2 based reward model that outputs a scalar value per sequence.
    """
    def __init__(self, base_model="gpt2"):
        super().__init__()
        config = GPT2Config.from_pretrained(base_model)
        self.transformer = GPT2Model.from_pretrained(base_model, config=config)
        self.value_head = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        Returns a scalar reward per sequence in the batch: (batch,)
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # For simplicity, let's just pool by taking the last token's hidden state
        pooled_output = last_hidden_state[:, -1, :]  # (batch, hidden_dim)

        reward = self.value_head(pooled_output).squeeze(-1)  # (batch,)
        return reward
