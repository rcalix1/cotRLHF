## Author: Ricardo A. Calix, Ph.D.
## Last update Feb 8, 2025
## Released as is with no warranty
## MIT License

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import sklearn
import random
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
## coefficient of determination 
## from sklearn.metrics import r2_score
## from einops import rearrange
from math import sqrt, log
torch.manual_seed(256)
from datetime import datetime
## import similaritymeasures



############################################################

class trainingRLHF:

    def __init__(self):
        self.MyName            = 'training RLHF'
        self.model_type        = "gpt2"  ## "llama"  # Change to "gpt2" for GPT models
        self.mixed_precision   = True
        self.model_name_llama  = "meta-llama/Llama-2-7b-hf"
        self.model_name        = "distilgpt2"
        ## self.model_name     = "meta-llama/Llama-3.2-1B-Instruct"
        ## self.model_name     = "Qwen/Qwen2.5-1.5B-Instruct"
        self.vocab_size        = 100     # Small vocab for synthetic data
        self.embed_size        = 128
        self.num_heads         = 4
        self.num_layers        = 2
        self.hidden_dim        = 256
        self.max_seq_len       = 32
        self.seq_len           = 16
        self.batch_size        = 32
        self.epochs            = 10
        self.lr                = 1e-3
        self.reward_fn         = None

    #########################################################
    ## Fine-Tuning
    def fine_tune(self, model, data, epochs=3, lr=1e-4):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            for x, y in data:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


    #########################################################
    ## SFT -> Supervised Fine Tuning
    def supervised_fine_tuning(self, dataset, num_epochs=1, batch_size=2, learning_rate=5e-5):
        dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True )
        optimizer  = optim.AdamW( self.model.parameters(), lr=learning_rate )
        self.model.train()
        for epoch in range( num_epochs ):
            total_loss = 0
            for batch in dataloader:
                inputs = self.tokenizer(batch["input"],  return_tensors="pt", padding=True, truncation=True).to("cuda")
                labels = self.tokenizer(batch["output"], return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
                labels[ labels == self.tokenizer.pad_token_id ] = -100
                outputs = self.model(**inputs, labels=labels)
                loss    = outputs.loss       ## ??
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    #########################################################
    def fine_tune_with_rl( self, prompts, expected_answers, num_epochs=1, batch_size=2, learning_rate=1e-5, clip_epsilon=0.2 ):
        optimizer = optim.AdamW(  model.parameters(), lr=learning_rate  )
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(0, len(prompts), batch_size):
                batch_prompts   = prompts[         i:i + batch_size ]
                batch_answers   = expected_answers[i:i + batch_size ]
                generated_texts = [ self.generate(prompt) for prompt in batch_prompts ]
                rewards = torch.tensor([
                    self.reward_fn( output, expected ) for output, expected in zip(generated_texts, batch_answers)
                ], dtype=torch.float32).to("cuda")
                old_log_probs = []    ## Compute old and new log probabilities
                new_log_probs = []
                for prompt, generated_text in zip(batch_prompts, generated_texts):
                    
                    inputs            = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                    outputs           = self.model(**inputs, labels=inputs.input_ids)
                    old_log_probs.append( outputs.logits.mean().detach() )
                    
                    generated_inputs  = self.tokenizer(generated_text, return_tensors="pt").to("cuda")
                    generated_outputs = self.model(**generated_inputs, labels=generated_inputs.input_ids)
                    new_log_probs.append( generated_outputs.logits.mean() )
                    
                old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to("cuda")
                new_log_probs = torch.tensor(new_log_probs, dtype=torch.float32).to("cuda")
                
                loss = self.compute_grpo_loss(old_log_probs, new_log_probs, rewards, clip_epsilon)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(prompts):.4f}")

    #########################################################
    ## DPO
    def train_dpo(self, gpt_model, reward_model, optimizer_gpt, optimizer_reward, vocab_size, seq_len, epochs, batch_size):
    
        for epoch in range(epochs):
            seq_a, seq_b, preferences = generate_synthetic_data(batch_size, seq_len, vocab_size)
            logits_a = gpt_model(seq_a)
            logits_b = gpt_model(seq_b)
            reward_a = reward_model(logits_a.mean(dim=1))
            reward_b = reward_model(logits_b.mean(dim=1))

            loss_dpo_gpt = dpo_loss(reward_a, reward_b, preferences)
            optimizer_gpt.zero_grad()
            loss_dpo_gpt.backward()
            optimizer_gpt.step()

            # Recompute logits for the Reward Model update
            logits_a = gpt_model(seq_a).detach()  # Detach to avoid tracking gradients for GPT again
            logits_b = gpt_model(seq_b).detach()

            # Forward pass through the reward model
            reward_a = reward_model(logits_a.mean(dim=1))
            reward_b = reward_model(logits_b.mean(dim=1))

            # Calculate DPO loss and backpropagate for reward model
            loss_dpo_reward = dpo_loss(reward_a, reward_b, preferences)
            optimizer_reward.zero_grad()
            loss_dpo_reward.backward()
            optimizer_reward.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss (GPT): {loss_dpo_gpt.item()}, Loss (Reward): {loss_dpo_reward.item()}")



    #########################################################
    def printName(self):
        print( self.MyName  )

    




