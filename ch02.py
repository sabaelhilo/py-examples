import json
from collections import Counter
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/saba/Documents/GitHub/pydata-book/ch02/usagov_bitly_data2012-03-16-1331923249.txt'
records = [json.loads(line) for line in open(path)]

# most often occurring timezone in the data
timezones_with_empty = [r.get('tz', '') for r in records]
timezones_with_check = [r['tz'] for r in records if 'tz' in r]

filter_unknown = [t for t in timezones_with_empty if t != '']

# count timezones
# using python
# go through list and have a dict/map with counts


def count_timezones_1(timezone_list):
    counts_dict = {}
    for tz in timezone_list:
        count = counts_dict.get(tz, 0) + 1
        counts_dict[tz] = count
    return counts_dict


def top_counts(timezone_dict, n=10):
    value_key_pairs = [(value, key) for key, value in timezone_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

# using the counters collection

counts = Counter(filter_unknown)
counts.most_common(10)

# using pandas
frame = DataFrame(records)

# how to count number of rows of a value (series)
tz_counts = frame['tz'].value_counts()
top = tz_counts[:10]

# replace na with missing and '' with unknown
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()

# plotting data
tz_counts[:10].plot(kind='barh', rot=0)

# parsing the agent information
agent_info = Series([r.split()[0] for r in frame.a.dropna()])
agent_counts = agent_info.value_counts()
top_agent_counts = agent_counts[:10]

top_agent_counts.plot(kind='barh', rot=0)

# want to show the top timezones into windows and non windows users
# windows if the agent string contains windows
agent_frame = frame[frame.a.notnull()]
operating_system = np.where(agent_frame['a'].str.contains('Windows'), 'Windows', 'Not Windows')
by_tz_os = agent_frame.groupby(['tz', operating_system])
reshape_agg_counts = by_tz_os.size().unstack().fillna(0)

# top overall time zones - sum(1) adds windows and non windows into one value
# arg sort returns the indicies that would sort an array
indexer = reshape_agg_counts.sum(1).argsort()
count_subset = reshape_agg_counts.take(indexer)[-10:]

count_subset.plot(kind='barh', stacked=True)

# get percentage of users for each time zone
normalized_subset = count_subset.div(count_subset.sum(1), axis=0)
normalized_subset.plot(kind='barh', stacked=True)

# movielens 1M data set
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('/Users/saba/Documents/ml-1m/users.dat', sep='::', header=None, names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('/Users/saba/Documents/ml-1m/ratings.dat', sep='::', header=None, names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('/Users/saba/Documents/ml-1m/movies.dat', sep='::', header=None, names=mnames)

# merge all data into one table
data = pd.merge(pd.merge(ratings, users), movies)

# get mean movie ratings for each film grouped by gender
# title is the rows, gender is the columns and rating is the values
mean_ratings = pd.pivot_table(data, 'rating', 'title', 'gender', aggfunc='mean')


# filter down movies that recieved at least 250 ratings
count_by_title = data.groupby('title').size()
active_titles = count_by_title.index[count_by_title >= 250]

# the index of titles from above can be used to select rows from mean_ratings
mean_ratings = mean_ratings.loc[active_titles]

# see top ratings among female viewers
top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)

# find movies that are most divisive between male and female viewers
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')
preferred_by_women = sorted_by_diff[:15]

# reverse order to get by men
reverse = sorted_by_diff[::-1][:15]

# find movies that elicited the most disagreement between viewers - user variance or standard deviation
# standard deviation of rating grouped by title
rating_std_by_title = data.groupby('title')['rating'].std()

# filter to active titles
rating_std_filtered = rating_std_by_title.loc[active_titles]

#sort series
rating_std_filtered.sort_values(ascending=False)

# US baby names
names1880 = pd.read_csv('/Users/saba/Documents/names/yob1880.txt', names=['name', 'sex', 'births'])

# get total number of births by gender
sum_by_gender = names1880.groupby('sex')['births'].sum()

# get all years of data and add a year column
years = range(1880, 2015)
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    path = '/Users/saba/Documents/names/yob{}.txt'.format(year)
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)

names = pd.concat(pieces, ignore_index=True)

# aggregate data on the year and sex level
# total births by year and gender
total_births = pd.pivot_table(names,'births', 'year', 'sex', aggfunc=sum)
total_births.plot(title='Total births by year and gender')

# fraction of babies given each name relative to total births


def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births / births.sum()
    return group

names = names.groupby(['year', 'sex']).apply(add_prop)

# check if close to 1
np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)

# extract top 1000 names for each sex/year combination


def get_top_1000(group):
    sorted_values = group.sort_values(by='births', ascending=False)
    return sorted_values[:1000]

grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top_1000)
top1000.index = np.arange(len(top1000))

boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']

# pivot table by total number of births by year and name
total_births = pd.pivot_table(top1000, 'births', 'year', 'name', aggfunc=sum)
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False, title="Number of births per year")

table = pd.pivot_table(top1000, 'prop', 'year', 'sex', aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex', yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))

# number of distinct names taken in order of popularity from highest to lowest in the top 50% births

df = boys[boys.year == 2010]
prop_cumsum = df.sort_values(by='prop', ascending=False).prop.cumsum()
prop_cumsum.values.searchsorted(0.5)


def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(q) + 1
diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
diversity.plot(title = "Number of popular names in top 50%")

get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'
names['last_letters'] = last_letters

table = pd.pivot_table(names, 'births', 'last_letters', ['sex', 'year'], aggfunc=sum)
subtable = table.reindex(columns = [1910, 1960, 2010], level='year')
letter_prop = subtable / subtable.sum().astype(float)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female')

letter_prop = table / table.sum().astype(float)
dny_ts = letter_prop.loc[['d', 'n', 'y'], 'M'].T
dny_ts.plot()

all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]

filtered = top1000[top1000.name.isin(lesley_like)]
filtered.groupby('name').births.sum()

plt.show()
