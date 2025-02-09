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

class RewardModel(  nn.Module  ):
    
    def __init__(self, embed_size, hidden_dim):
        
        super(RewardModel, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embeddings):
        
        return self.fc( embeddings ).squeeze(-1)

############################################################

class COTrewards:

    def __init__(self):
        self.MyName            = 'Rewards COT'

    ##########################
    def printName(self):
        print( self.MyName  )

    ###################################################################
    def correctness_reward_func(  self, prompts, completions, answer ):
    
        responses           = [  completion[0]['content'] for completion in completions  ]
        q                   =    prompts[0][-1]['content']
        extracted_responses = [  extract_xml_answer(r) for r in responses  ]
        print('-'*20, 
              f"Question:\n{q}", 
              f"\nAnswer:\n{answer[0]}", 
              f"\nResponse:\n{responses[0]}", 
              f"\nExtracted:\n{extracted_responses[0]}")
        
        return [  2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)  ]


    #################################################
    def int_reward_func( self, completions ) :
        
        responses           = [ completion[0]['content'] for completion in completions ]
        extracted_responses = [ extract_xml_answer(r) for r in responses ]
        return [ 0.5 if r.isdigit() else 0.0 for r in extracted_responses ]

    ######################################################################
    def strict_format_reward_func( self, completions ):
        
        """Reward function that checks if the completion has a specific format."""
        pattern   = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [ completion[0]["content"] for completion in completions ] 
        matches   = [ re.match(pattern, r) for r in responses ] 
        return [ 0.5 if match else 0.0 for match in matches ]

    
    #####################################################################
    def soft_format_reward_func( self, completions ):
        """Reward function that checks if the completion has a specific format."""
        pattern   = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [ completion[0]["content"] for completion in completions ]
        matches   = [ re.match(pattern, r) for r in responses ] 
        return [  0.5 if match else 0.0 for match in matches  ]


    ##############################################################
    def xmlcount_reward_func( self, completions ):
        
        contents = [ completion[0]["content"] for completion in completions ]
        return [ self.count_xml(c) for c in contents ]

    ##############################################################
    def extract_answer( self, output_text ):
    
        if "[answer]" in output_text and "[/answer]" in output_text:
            start = output_text.find("[answer]") + len("[answer]")
            end   = output_text.find("[/answer]")
            return output_text[start:end].strip()
        return None

    
    ##############################################################
    def count_xml( self, text ):
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1])*0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
        return count

    #############################################
    def extract_hash_answer( self, text ):
        
        if "####" not in text:
            return None
        return text.split("####")[1].strip().replace(",", "").replace("$", "")

    #############################################
    # Rule-based Reward Function for Multi-Step Reasoning

    def rule_based_reward(output_text, expected_answer=None, task_type="reasoning"):
    
        reward = 0.0

        # Format Reward: Check for proper reasoning structure
        if "<think>" in output_text and "</think>" in output_text:
            reward += 0.3  # Reward for using the correct format

        # Step-by-Step Reward: Check intermediate steps
        steps = [segment.strip() for segment in output_text.split("<think>") if "</think>" in segment]
        for step in steps:
            if step in expected_answer:  # Check if the step matches the expected reasoning
                reward += 0.2 / len(steps)  # Reward each correct step proportionally

        # Final Answer Reward: Check for correct answer
        if "[answer]" in output_text and "[/answer]" in output_text:
            answer = extract_answer(output_text)
            if answer == extract_answer(expected_answer):
                reward += 0.5

        return reward

   

    #############################################
    def extract_xml_answer( self, text ):
        
        answer =   text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()



        
