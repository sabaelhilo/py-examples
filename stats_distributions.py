import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series
from scipy import stats

# baby boom data
def eval_cdf(x, samples):
    count = 0
    for s in samples:
        if s <= x:
            count += 1
    return count / len(samples)

names = ['time', 'gender', 'weights', 'minutes']
baby_boom = pd.read_table('/Users/saba/Documents/babyboom.dat', header=None, names=names, sep='\s+')
baby_boom_minute_diff = baby_boom.minutes.diff()
cdf_diff = [eval_cdf(d, baby_boom_minute_diff) for d in baby_boom_minute_diff]
plt.figure(1)
plt.suptitle('Baby boom cdf (diff)')
plt.plot(list(baby_boom_minute_diff.values), cdf_diff, 'ro')

cdf_minutes = [eval_cdf(d, baby_boom.minutes) for d in baby_boom.minutes]
plt.figure(2)
plt.suptitle('Baby boom minutes cdf')
plt.plot(list(baby_boom.minutes.values), cdf_minutes, 'ro')

# the normal distribution
baby_boom_mean = baby_boom.weights.mean()
baby_boom_std = baby_boom.weights.std()
baby_boom_weight_cdf = [eval_cdf(w, baby_boom.weights) for w in baby_boom.weights]
plt.figure(4)
plt.plot(list(baby_boom.weights.values), baby_boom_weight_cdf, 'ro')
plt.suptitle('Baby boom weight cdf')

normal_dist = stats.norm.cdf(baby_boom.weights.values, loc=baby_boom_mean, scale=baby_boom_std)
# returns the value of cdf at the values passed, so creates a normal distribution from log, scale and applies
# it to values

plt.figure(5)
plt.suptitle('Baby boom weight normal distribution')
plt.plot(list(baby_boom.weights.values), normal_dist, 'bo')

# create a frozen gaussian distribution
n = stats.norm(loc=3.5, scale=2.0)
n.rvs()  # draw a random sample

# Normal probability plot - to test if a normal distribution is a good fit for the data
plt.figure(6)
sorted_weights = baby_boom.weights.sort_values()
plt.plot(sorted_weights.values, 'o')
plt.suptitle('Sorted baby weights')

sample_size = len(sorted_weights)
s = np.random.normal(0, 1, len(sorted_weights))
s.sort()
plt.figure(7)
plt.plot(Series(s).values, 'o')


# lognormal - if the logs of values have a normal distribution

plt.show()
