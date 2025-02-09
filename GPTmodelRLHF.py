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

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM


############################################################

class GPT_HF( nn.Module ):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, max_seq_len):
        super(GPT_HF, self).__init__()
        self.embedding           = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_size))
        self.transformer         = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads, hidden_dim),
            num_layers
        )
        self.fc                  = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer( x )
        return self.fc(       x )


############################################################

class Block(nn.Module):
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ff   = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

############################################################


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head             = n_head
        self.head_dim           = n_embd // n_head
        self.query              = nn.Linear(n_embd, n_embd)
        self.key                = nn.Linear(n_embd, n_embd)
        self.value              = nn.Linear(n_embd, n_embd)
        self.proj               = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)))

    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(  B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

############################################################

class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

############################################################
## GPT Architecture (compatible with pre-trained weights)

class GPT(nn.Module):
    
    def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks             = nn.ModuleList([
            Block(n_embd, n_head) for _ in range(n_layer)
        ])
        self.ln_f               = nn.LayerNorm(n_embd)
        self.head               = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        
        B, T    = idx.size()
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(  torch.arange(T, device=idx.device)  )
        x       = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x       = self.ln_f(x)
        logits  = self.head(x)
        return logits




############################################################

class GPTmodelRLHF:

    def __init__(self):
        self.MyName            = 'GPT model RLHF'

        self.GPTmodel          = None

    #########################################################
    def printName(self):
        print( self.MyName  )

    #########################################################
    def load_some_GPT_from_HF_hub(self, model_type ="gpt2"):
        if model_type == "gpt2":
            self.tokenizer           = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Add padding token for GPT-2
            self.model = AutoModelForCausalLM.from_pretrained(
                   self.model_name, 
                   torch_dtype = torch.float16 if self.mixed_precision else torch.float32
            ).cuda()
        elif model_type == "llama":
            self.tokenizer = LlamaTokenizer.from_pretrained( self.model_name )
            with init_empty_weights():
                self.model = LlamaForCausalLM.from_pretrained(
                       model_name, 
                       torch_dtype=torch.float16 if self.mixed_precision else torch.float32
                )
                self.model = load_checkpoint_and_dispatch(
                    model, 
                    model_name, 
                    device_map = "auto", 
                    offload_folder = "offload"
                )
        else:
            raise ValueError("Unsupported model type. Use 'gpt2' or 'llama'.")

        model.gradient_checkpointing_enable()

    ##
        

    
    #########################################################
    def load_GPT2_pre_trained_weights(self):

        self.model_name = "gpt2"
        self.tokenizer  = GPT2Tokenizer.from_pretrained(self.model_name)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

        # Access weights
        self.gpt2_weights = self.gpt2_model.state_dict()
        
        # Initialize Custom GPT Model
        self.vocab_size = self.gpt2_weights["transformer.wte.weight"].shape[0]
        self.block_size = self.gpt2_model.config.n_ctx
        self.n_embd     = self.gpt2_model.config.n_embd
        self.n_layer    = self.gpt2_model.config.n_layer
        self.n_head     = self.gpt2_model.config.n_head

        self.model      = self.GPT_HF(vocab_size, block_size, n_embd, n_layer, n_head)

        # Map GPT-2 weights to custom GPT model
        self.model.token_embedding.weight.data    = self.gpt2_weights["transformer.wte.weight"].clone()
        self.model.position_embedding.weight.data = self.gpt2_weights["transformer.wpe.weight"].clone()

    #########################################################
    def generate( self, input_text, max_length=50 ):
        inputs  = tokenizer( input_text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs.input_ids, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode( outputs[0], skip_special_tokens=True )


    #########################################################
    def evaluate( self, prompts, expected_answers ):
        self.model.eval()
        correct_answers   = 0
        correct_reasoning = 0

        for prompt, expected in zip( prompts, expected_answers ):
            output = self.generate( prompt)
            if self.extract_answer( output ) == self.extract_answer( expected ):
                correct_answers += 1

            steps = [ segment.strip() for segment in output.split("<think>") if "</think>" in segment ]
            expected_steps = [ segment.strip() for segment in expected.split("<think>") if "</think>" in segment ]
            correct_reasoning += sum( 1 for step in steps if step in expected_steps )

        total_prompts      = len( prompts )
        reasoning_accuracy = correct_reasoning / total_prompts
        answer_accuracy    =   correct_answers / total_prompts

        return {
            "answer_accuracy": answer_accuracy,
            "reasoning_accuracy": reasoning_accuracy,
        }

    #########################################################
    def extract_answer( self, output_text ):
    
        if "[answer]" in output_text and "[/answer]" in output_text:
            start = output_text.find("[answer]") + len("[answer]")
            end   = output_text.find("[/answer]")
            return output_text[start:end].strip()
        return None


    #########################################################
    









        

    