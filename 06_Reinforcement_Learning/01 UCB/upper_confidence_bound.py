# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Initialising
n = 10000
d = 10
ads_selected = []
total_reward = 0

"""
# Random selection
import random

for n in range(0, n):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward += reward
    accuracy = total_reward/n
    """

#UCB
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
for i in range(0, n):
    ad = 0
    max_upper_bound = 0
    for j in range(0, d):
        if(numbers_of_selections[j] > 0):
            avg_reward = sums_of_rewards[j]/numbers_of_selections[j]
            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[j])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = j
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[i, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
    
for i in range (0, d):
    total = []
    total = dataset.sum(axis = 0)
    
#Visualising results
plt.hist(ads_selected)
plt.title('Ads Selected')
plt.xlabel('Ads')
plt.ylabel('Times each ad was selected')
plt.show()
