import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy import stats
import math

# movielens 1M data set
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('/Users/saba/Documents/ml-1m/users.dat', sep='::', header=None, names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('/Users/saba/Documents/ml-1m/ratings.dat', sep='::', header=None, names=rnames)

# mnames = ['movie_id', 'title', 'genres']
# movies = pd.read_table('/Users/saba/Documents/ml-1m/movies.dat', sep='::', header=None, names=mnames)

# merge all data into one table
movie_data = pd.merge(ratings, users)
age_data = movie_data.age
age_hist_data = age_data.value_counts()

age_data.hist(grid=False, bins=20)

# remove data with age 1
filtered_age_data = movie_data[age_data > 1]

# summary stats
mean_age = filtered_age_data.age.mean()
std_age = filtered_age_data.age.std()

group1 = movie_data[movie_data.occupation == 6]
group2 = movie_data[movie_data.occupation == 7]

# effect size of two groups cohen's d
mean_diff = group1.age.mean() - group2.age.mean()
n1, n2 = len(group1), len(group2)
pooled_var = (n1 * group1.age.var() + n2 * group2.age.var()) / (n1 + n2)
d_cohen = mean_diff / math.sqrt(pooled_var)  # difference in means is 0.2 standard deviations

# probabilities
sum_age = movie_data.age.value_counts()
percentage = sum_age / sum_age.values.sum()
movie_data['age_prob'] = percentage[movie_data.age].values

group1_prob = movie_data[movie_data.occupation == 6].age_prob
group2_prob = movie_data[movie_data.occupation == 7].age_prob
z = np.zeros(len(group2_prob))
z[0:len(group1_prob)] = group1_prob
group_data = DataFrame({'group1': z, 'group2': group2_prob})
group_data.hist(grid=False, bins=15)

# cdf cumulative distribution function - maps from a value to its percentile rank
# to evaluate cdf(x) we have to evaluate the fraction of values that are x or less than x - cdf is a step function


def eval_cdf(x, samples):
    count = 0
    for s in samples:
        if s <= x:
            count += 1
    return count / len(samples)

group_data.group1[group_data['group1'] == 0] = np.nan
group_data.quantile(0.5)


# baby boom data
names = ['time', 'gender', 'weights', 'minutes']
baby_boom = pd.read_table('/Users/saba/Documents/babyboom.dat', header=None, names=names, sep='\s+')
baby_boom_minute_diff = baby_boom.minutes.diff()
cdf_diff = [eval_cdf(d, baby_boom_minute_diff) for d in baby_boom_minute_diff]
plt.figure(3)
plt.plot(list(baby_boom_minute_diff.values), cdf_diff, 'ro')

cdf_minutes = [eval_cdf(d, baby_boom.minutes) for d in baby_boom.minutes]
plt.figure(4)
plt.plot(list(baby_boom.minutes.values), cdf_minutes, 'ro')

# normal distribution
baby_boom_mean = baby_boom.weights.mean()
baby_boom_std = baby_boom.weights.std()
plt.figure(5)
# plt.plot(baby_boom.weights, 'ro')
baby_boom.weights.hist(grid=False, bins=15)
baby_boom_weight_cdf = [eval_cdf(w, baby_boom.weights) for w in baby_boom.weights]
plt.figure(6)
plt.plot(list(baby_boom.weights.values), baby_boom_weight_cdf, 'ro')

normal_dist = stats.norm.cdf(baby_boom.weights.values, loc=baby_boom_mean, scale=baby_boom_std)
plt.plot(list(baby_boom.weights.values), normal_dist, 'bo')

plt.show()
