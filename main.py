
import os
import numpy as np
import random
import pickle as pkl
import re
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from preprocessing import ReadTweet
FILEPATH = "data/eit_annot_100_1899.txt"

# Get tweets
X, y = ReadTweet(FILEPATH).get_data()
n_tweets = y.shape[0]

