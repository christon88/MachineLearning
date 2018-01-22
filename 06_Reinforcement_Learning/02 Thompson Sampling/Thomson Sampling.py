# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Initialising
n = 10000
d = 10
ads_selected = []
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
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

#Thomson Sampling

for i in range(0, n):
    ad = 0
    max_random = 0
    for j in range(0, d):
        #Generate random draws
        random_beta = random.betavariate(number_of_rewards_1[j] + 1, number_of_rewards_0[j] + 1)
        
        if random_beta > max_random:
             max_random = random_beta
             ad = j
    ads_selected.append(ad)
    reward = dataset.values[i, ad]
    if reward == 1:
        number_of_rewards_1[ad] += 1
    else:
        number_of_rewards_0[ad] += 1
        
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
