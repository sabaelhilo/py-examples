import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from pandas_datareader import data

# series - one dimensional array and an array of data label - index
obj = Series([4, 7, -5, 3])

index = obj.index
values = obj.values

obj2 = Series([4, 7, -5, 3], index=['a', 'b', 'c', 'd'])
a = obj2['a']
# pandas fancy indexing
subset = obj2[['c', 'a', 'd']]

# numpy array operations work on pandas series
# they are like a fixed length ordered dict
greater_than_zero = obj2[obj2 > 0]
b_in = 'b' in obj2
sdata = {'Ohio': 35000, 'Texas': 7000}
obj3 = Series(sdata)
new_states = {'California', 'Ohio'}
obj4 = Series(sdata, index=new_states)  # California NaN  Ohio          35000.0

pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()

obj4.name = 'population'
obj4.index.name = 'state'
obj4.index = ['Bob', 'Square']

# dataFrame - a tabular spreadsheet data structure  (dict of series)
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

frame = DataFrame(data, columns=['year', 'state', 'pop'], index=['one', 'two', 'three', 'four', 'five'])

state_column = frame['state']
state_column_attr = frame.state

third_row_loc = frame.loc[2]
frame['debt'] = np.arange(5)

frame['eastern'] = frame.state == 'Ohio'
del frame['eastern']

frame.index.name = 'number'
frame.columns.name = 'state'

vals = frame.values

obj = Series(range(3), index=['a', 'b', 'c'])

# essential functionality
# 1. reindex - new obj with data conformed to new index
obj1 = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj1.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)  # replaces NA w/ 0

obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(np.arange(6), method='ffill')

# reindex rows and columns
frame1 = DataFrame(data)
frame1.reindex([0, 1, 2], columns=['pop', 'debt', 'year'])
frame1.loc[[0, 1, 2], ['pop', 'debt', 'year']]

obj4 = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj4 = obj4.drop(['d', 'c'])

frame1.drop([0, 1])
frame1.drop('state', axis=1)

row_1_col = frame1.loc[1, ['pop', 'year']]
row_1 = frame1.loc[1]
rows_state_col = frame1.loc[:3, 'state']
entries_above_2000 = frame1.loc[frame1.year > 2000]

frame1.set_value(1, 'state', 'cali')

# arithmetic and data alignment
# broadcasting - operations between dataframes and series
frame2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series_first_row = frame2.loc[0]
broadcast_rows = frame2 - series_first_row

series_first_col = frame2.b
broadcast_columns = frame2.sub(series_first_col, axis=0)

# applying function of 1D arrays to each column or row
# applies f to each column
frame3 = DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
f = lambda x: x.max() - x.min()
frame3.apply(f)
frame3.apply(f, axis=1)  # applies to rows


def f1(x):
    return Series([x.min(), x.max()], index=['min', 'max'])

frame3.apply(f1)

# element wise functions of dataframes
format_l = lambda x: '{}'.format(x)
frame3.applymap(format_l)
frame3['e'].map(format_l)

# sorting and ranking
obj5 = Series(range(4), index=['d', 'a', 'b', 'c'])
obj5.sort_index()

frame4 = DataFrame(np.arange(8).reshape(2, 4), index=['three', 'one'], columns=['d', 'a', 'b', 'c'])
frame4.sort_index()
frame4.sort_index(axis=1)  # sorts columns
frame4.sort_index(axis=1, ascending=False)

# sort series by its values
obj6 = Series([4, 7, -3, 2])
obj6.order()


# sort data frame by its values
frame5 = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame5.sort_index(by='b')
frame5.sort_index(by=['a', 'b'])

# summarizing and computing descriptive stats
frame6 = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'],
                   columns=['one', 'two'])
frame6_column_sum = frame6.sum()
frame6_row_sum = frame6.sum(axis=1, skipna=False)

frame6.idxmax()  # index of maximum value
frame6.cumsum()
frame6.describe()

# correlation and covariance
all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT']:
    all_data[ticker] = data.get_data_yahoo(ticker)

price = DataFrame({tic: tic_data['Adj Close'] for tic, tic_data in all_data.items()})
volume = DataFrame({tic: tic_data['Volume'] for tic, tic_data in all_data.items()})

percent_chng = price.pct_change()

# stats on series - returns one value
percent_chng.MSFT.corr(percent_chng.IBM)
percent_chng.MSFT.cov(percent_chng.IMB)

# stats on df - returns full matrix
percent_chng.corr()
percent_chng.cov()
percent_chng.corrwith(percent_chng.IBM)
percent_chng.corrwith(volume)

# unique values, value counts and membership
obj7 = Series(['c', 'a', 'd', 'a', 'a'])
uni = obj7.unique()
uni.sort()
obj7.value_counts()  # value frequencies

mask = obj7.isin(['b', 'c'])

# build histogram from data
frame7 = DataFrame({'Qu1': [1, 3, 4, 3, 4], 'Qu2': [2, 3, 1, 2, 3], 'Qu3': [1, 5, 2, 4, 4]})
frame7.apply(pd.value_counts)

# missing data
string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data.isnull()
string_data[0] = None

obj8 = Series([1, np.nan, 3.5, np.nan, 7])
obj8.dropna()

frame8 = DataFrame([[1., 6.5, 3], [1., np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, 6.5, 3]])
cleaned_frame8 = frame8.dropna()  # drops whole row containing NA value
frame8.dropna(how='all')

# hierarchical indexing - multiple index levels on an axis
obj9 = Series(np.random.randn(6), index=[['a', 'a', 'b', 'b', 'c', 'c'], [1,2,1,2,1,2]])
obj9['a'][1]
obj9.unstack()
