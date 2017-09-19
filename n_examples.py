import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr

# numpy chapter (ch04)
# ndarray - all elements have to be the same type

data = [6, 7.5, 8, 0, 1]

arr = np.array(data)

data2 = [[1, 2, 3, 4], [8, 9, 19, 11]]
arr2 = np.array(data2)
# array([1, 2, 3, 4], [8, 9, 19, 11)

# property operations on numpy arrays
dim = arr2.ndim
sh = arr2.shape
ty = arr2.dtype

# numpy array declarations
arr = np.array(data)
zero = np.zeros(2)  # array([0, 0])
zero_mat = np.zeros((2, 2))  # array([0, 0], [0, 0])
empty_3d_mat = np.empty((2, 3, 2))  # can return zeros or unintialized garbage values

# numpy array functions
ar = np.arange(15)  # array([0, 1, 2, ..., 15])

# numpy data types
arr1 = np.array([1, 2, 3, 4])
arr1.dtype  # int64
float_arr1 = arr1.astype(np.float64)

numeric_strings = np.array(['1.25', '-9.6', '42'])
numeric_strings.astype(float)  # array([1.35, -9.6 ...

# vectorization - express batch operations without writing for loops
# arithmetic operations between equal size arrays applies the operation elementwise
arr3 = np.array([[1, 2, 3], [4, 5, 6]])
arr3.astype(np.float64)
multi = arr3 * arr3
minus = arr3 - arr3

# arithmetic operations with scalars propagate the value to each element
div = 1 / arr3

# basic indexing and slicing
# array slices are views on the original array, data is not copied any modifications
# are reflected to the source
arr4 = np.arange(10)
arr_slice = arr4[0:2]
arr_slice[1] = 100
print(arr4) # [  0 100   2   3   4   5   6   7   8   9]
arr_slice_copy = arr4[0:2].copy()

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d_slice = arr2d[2]  # [7, 8, 9]
arr2d_individual = arr2d[0][2]  # row 0, col 2 -> 3

arr3d = ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d_slice = arr3d[0]  # array([[1, 2, 3], [4 ,5, 6]])

# boolean indexing
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
bob_name = names == 'Bob'  # array([True, False, False, True ...
data_bob_row = data[names == 'Bob']  # returns row 0, 3
data_bob_row_col = data[names == 'Bob', :2]
mask = (names == 'Bob') | (names == 'Will')  # use | and & for opertations on arrays
data[names != 'Joe'] = 7

# fancy indexing - copies data
arr5 = np.empty((8, 4))

for i in range(8):
    arr5[i] = i

rows_jumbled_order = arr5[[4, 3, 0, 6]]
rows_from_end = arr5[[-3, -5, -7]]

arr6 = np.arange(32).reshape((8, 4))
select_row_columns = arr6[[1, 5, 7, 2], [0, 3, 1, 2]]  # this returns a 1d array where each element is (row, col)
select_row_all_columns = arr6[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]  # returns 2d array where its (row, [0, 3, 1, 2])

# transpose and axis swapping
arr7 = np.arange(15).reshape((3, 5))
arr7_t = arr7.T
dot_product = np.dot(arr7_t, arr7)

# universal functions
# unary ufuncs
arr8 = np.arange(10)
sqr = np.sqrt(arr8)
e = np.exp(arr8)

# binary ufuncs
x = np.random.randn(8)
y = np.random.randn(8)

np.maximum(x, y)  # elementwise maximum

# Data processing using arrays
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs ** 2 + ys ** 2)

# plots the result of sqrt(x^2 + y^2)
plt.imshow(z, cmap=plt.cm.gray)
plt.colorbar()

# expressing conditional logic as array operations
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

# take a value from xarr whenever the corresponding value in cond is true else take from yarr
val = [(x if z else y) for (x, y, z) in zip(xarr, yarr, cond)]

# where - produce a new array of values based on another array
result = np.where(cond, xarr, yarr)
mat = np.random.rand(4, 4)
np.where(mat > 0, 2, -2)

# methods for boolean arrays
arr8 = np.random.randn(100)
positive_values = (arr8 > 0).sum()
bools = np.array([True, False, False, True])
bools.any()  # checks if any of the values in array are true
bools.all()  # check if every value is true


# sorting
arr9 = np.random.randn(8)
arr9.sort()  # sorts the array

# compute quantiles
large_arr = np.random.randn(1000)
large_arr.sort()
five_percent_quantile = large_arr[int(0.05 * len(large_arr))]

# unique returns sorted unique values and set logic
names_again = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
unique_names = np.unique(names_again)

# Linear algebra
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x.dot(y)
np.dot(x, y)


x1 = np.random.randn(5, 5)
mat1 = x1.T.dot(x1)
in_mat1 = inv(mat1)
identity_mat = mat1.dot(in_mat1)
q, r = qr(mat1)

# random number generation
samples = np.random.normal(size=(4, 4))  # random numbers from normal distribution
n_x = np.arange(10)
n_y = np.random.normal(size=10)
plt.figure(2)
plt.plot(n_x, n_y)
plt.show()
