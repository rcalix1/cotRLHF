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

## import Time_Series_GPT as Time_Series_GPT

############################################################


data_multi_step_reasoning = [
    {
        "input": "Why is the sky blue?",
        "output": (
            "<think>Step 1: Sunlight contains all colors of light.</think> "
            "<think>Step 2: As sunlight passes through the atmosphere, it interacts with air molecules.</think> "
            "<think>Step 3: Shorter wavelengths, like blue, scatter more than longer wavelengths, like red.</think> "
            "[answer]Rayleigh scattering[/answer]"
        )
    },
    {
        "input": "What is 2+2?",
        "output": (
            "<think>Step 1: Start with the first number: 2.</think> "
            "<think>Step 2: Add the second number: 2.</think> "
            "<think>Step 3: The result of the addition is 4.</think> "
            "[answer]4[/answer]"
        )
    }
]

############################################################

prompts_multi_step_reasoning = ["Why is the sky blue?", "What is 2+2?"]

expected_answers_multi_step_reasoning = [
    (
        "<think>Step 1: Sunlight contains all colors of light.</think> "
        "<think>Step 2: As sunlight passes through the atmosphere, it interacts with air molecules.</think> "
        "<think>Step 3: Shorter wavelengths, like blue, scatter more than longer wavelengths, like red.</think> "
        "[answer]Rayleigh scattering[/answer]"
    ),
    (
        "<think>Step 1: Start with the first number: 2.</think> "
        "<think>Step 2: Add the second number: 2.</think> "
        "<think>Step 3: The result of the addition is 4.</think> "
        "[answer]4[/answer]"
    )
]




############################################################


class dataCOT_RLHF:

    def __init__(self):
        self.MyName                                = 'Data RLHF COT'
        self.data_multi_step_reasoning             = data_multi_step_reasoning
        self.prompts_multi_step_reasoning          = prompts_multi_step_reasoning 
        self.expected_answers_multi_step_reasoning = expected_answers_multi_step_reasoning
        

        ##  Was: data -- Dummy data: sequence input and target
        data_dummy = [( 
               torch.randint( 0, vocab_size, (4, block_size)), 
               torch.randint( 0, vocab_size, (4, block_size))
        )]
        SYSTEM_PROMPT = """
                        Respond in the following format:
                        <reasoning>
                        ...
                        </reasoning>
                        <answer>
                        ...
                        </answer>
                        """
        XML_COT_FORMAT = """\
                         <reasoning>
                           {reasoning}
                         </reasoning>
                         <answer>
                         {answer}
                         </answer>
                         """

    ########################################
    def printName(self):
        print( self.MyName  )

    ########################################
    def extract_hash_answer( self, text ):
        
        if "####" not in text:
            return None
        return text.split("####")[1].strip().replace(",", "").replace("$", "")


    #########################################################
    # uncomment middle messages for 1-shot prompting
    def get_gsm8k_questions(self, split = "train"):
        
        data = load_dataset('openai/gsm8k', 'main')[split]    # type: ignore
        data = data.map(lambda x: { # type: ignore
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
                #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
                #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
                #    answer="7"
                #)},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': self.extract_hash_answer(x['answer'])
        })                      # type: ignore
        return data             # type: ignore


    ########################################################################
    def generate_synthetic_data( self, batch_size, seq_len, vocab_size ):
    
        seq_a = torch.randint(0, vocab_size, (batch_size, seq_len))
        seq_b = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Randomly assign preferences (1 means seq_a preferred over seq_b, 0 otherwise)
        preferences = torch.randint(0, 2, (batch_size,))
        return seq_a, seq_b, preferences







        
