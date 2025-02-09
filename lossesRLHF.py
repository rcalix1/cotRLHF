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

class lossesRLHF:

    def __init__(self):
        self.MyName            = 'losses RLHF'

    #########################################################
    def printName(self):
        print( self.MyName  )

    
    #########################################################
    def compute_grpo_loss(self, old_log_probs, new_log_probs, rewards, clip_epsilon=0.2):
        """
        Compute the GRPO loss for policy optimization.
        Args:
            old_log_probs (torch.Tensor): Log probabilities from the old policy.
            new_log_probs (torch.Tensor): Log probabilities from the new policy.
            rewards (torch.Tensor): Rewards for the generated outputs.
            clip_epsilon (float): Clipping parameter for PPO-like stability.

        Returns:
            torch.Tensor: GRPO loss.
        """
        ratios         =  torch.exp(  new_log_probs - old_log_probs  )
        clipped_ratios =  torch.clamp( ratios, 1 - clip_epsilon, 1 + clip_epsilon )
        loss           =  -torch.min(ratios * rewards, clipped_ratios * rewards).mean()
        return loss


    #########################################################
    def dpo_loss( self, reward_a, reward_b, preferences, beta=0.1 ):
    
        logits = (reward_a - reward_b) / beta
        loss   = -torch.mean( preferences * torch.log_softmax(logits, dim=0))
        return loss














        
