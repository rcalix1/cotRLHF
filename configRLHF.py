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

class configRLHF:

    def __init__(self):
        self.MyName            = 'config RLHF'
        self.model_type        = "gpt2"  ## "llama"  # Change to "gpt2" for GPT models
        self.mixed_precision   = True
        self.model_name_llama  = "meta-llama/Llama-2-7b-hf"
        self.model_name        = "distilgpt2"
        ## self.model_name     = "meta-llama/Llama-3.2-1B-Instruct"
        ## self.model_name     = "Qwen/Qwen2.5-1.5B-Instruct"

    #########################################################
    def printName(self):
        print( self.MyName  )

    













        

    