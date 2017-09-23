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

# probabilities - pmf probability mass function
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
# to evaluate cdf(x) where x is any value in the distribution we compute the
# fraction of values in the distribution less than or equal to x - its a step function


def eval_cdf(x, samples):
    count = 0
    for s in samples:
        if s <= x:
            count += 1
    return count / len(samples)

vals = [1, 2, 2, 3, 5]

e = [eval_cdf(v, vals) for v in vals]

# e is 0.2, 0.6, 0.8, 1 can evaluate
# cdf for values that don't appear in the sample
# if x is greater than the largest value then cdf is 1

group_data.group1[group_data['group1'] == 0] = np.nan
group_data.quantile(0.5)

plt.show()
