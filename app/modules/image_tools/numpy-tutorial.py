# %%
import numpy as np

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('2nd element on 1st row: ', arr[0, 1])

# %%
import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr[0, 1, 2])

# %%
import numpy as np

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('Last element from 2nd dim: ', arr[1, -1])

# %%
import numpy as np
"""We pass slice instead of index like this: [start:end].

We can also define the step, like this: [start:end:step]."""
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5])

# %%
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[4:])

# %%
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[:4])

#%%
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[-3:-1])

#%%
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5:2])

#%%
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[::2])

#%%
import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[1, 1:4])

#%%
import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[0:2, 2])

#%%
import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[0:2, 1:4])
print(f"dinension: {arr.ndim}")

#%%
import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr.dtype)

# %%
import numpy as np
"""Create an array with data type string:"""
arr = np.array([1, 2, 3, 4], dtype='S')

print(arr)
print(arr.dtype)

# %%
"""Create an array with data type 4 bytes integer:"""
import numpy as np

arr = np.array([1, 2, 3, 4], dtype='i4')

print(arr)
print(arr.dtype)

# %%
"""A non integer string like 'a' can not be converted 
to integer (will raise an error):"""

import numpy as np

arr = np.array(['a', '2', '3'], dtype='i')

# %%
"""Change data type from float to integer by using 'i' 
as parameter value:"""
import numpy as np

arr = np.array([1.1, 2.1, 3.1])

newarr = arr.astype('i')
newarr = arr.astype(int)

print(newarr)
print(newarr.dtype)

# %%
"""Make a copy, change the original array, and 
display both arrays:"""
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42

print(f"arr: {arr}")
print(f"x: {x}")

# %%
"""Make a view, change the original array, and display 
both arrays:"""
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42

print(f"arr: {arr}")
print(f"x: {x}")

# %%
"""Print the value of the base attribute to check if an 
array owns it's data or not:"""
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

x = arr.copy()
y = arr.view()

print(f"x from: {x.base}")
print(f"y from: {y.base}")

# %%
"""Print the shape of a 2-D array:"""
import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(arr.shape)
# %%
import numpy as np

arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('shape of array :', arr.shape)
# %%
"""Convert the following 1-D array with 12 elements into a 2-D array.
The outermost dimension will have 4 arrays, each with 3 elements:"""
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(4, 3)

print(newarr)
# %%
"""Try converting 1D array with 8 elements to a 2D array 
with 3 elements in each dimension (will raise an error):"""
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

newarr = arr.reshape(3, 3)

print(newarr)
# %%
"""Check if the returned array is a copy or a view:"""
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
"""Returns the original array, so it is a view."""
print(arr.reshape(2, 4).base)
# %%
"""Convert 1D array with 8 elements to 3D array with 2x2 elements:"""
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

newarr = arr.reshape(2, 2, -1)

print(newarr)
# %%
"""Convert the array into a 1D array:"""
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

newarr = arr.reshape(-1)

print(newarr)
# %%
"""Iterate on each scalar element of the 2-D array:"""
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  for y in x:
    print(y)
# %%
"""Iterate through the following 3-D array:"""
import numpy as np

arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for x in np.nditer(arr):
  print(x)
# %%
"""Iterate through the array as a string:"""
import numpy as np

arr = np.array([1, 2, 3])

for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x)
# %%
"""Iterate through every scalar element of the 2D array skipping 1 element:"""
import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for x in np.nditer(arr[:, ::2]):
  print(x)
# %%
"""Enumerate on following 2D array's elements:"""
import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for idx, x in np.ndenumerate(arr):
  print(idx, x)
# %%
"""Join two 2-D arrays along rows (axis=1):"""
import numpy as np

arr1 = np.array([[1, 2], [3, 4]])

arr2 = np.array([[5, 6], [7, 8]])

arr = np.concatenate((arr1, arr2), axis=1)

print(arr)
# %%
"""Joining Arrays Using Stack Functions
Stacking is same as concatenation, the only difference is that stacking is done along a new axis.
"""
import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)

print(arr)
# %%
"""NumPy provides a helper function: hstack() to stack along rows."""
import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.hstack((arr1, arr2))

print(arr)
# print(arr1+arr2)
# %%
"""NumPy provides a helper function: vstack()  to stack along columns."""
import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.vstack((arr1, arr2))

print(arr)
# %%
"""NumPy provides a helper function: dstack() to stack along height, which is the same as depth."""
import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.dstack((arr1, arr2))

print(arr)
# %%
"""Split the array in 4 parts: Returns a list"""
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 4)

print(newarr)
# %%
"""Split Into Arrays: Returns an array"""
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)

print(newarr)
print(newarr[1])
# %%
"""Split the 2-D array into three 2-D arrays."""
import numpy as np

arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

newarr = np.array_split(arr, 3)

print(newarr)
# %%
"""Split the 2-D array into three 2-D arrays."""
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.array_split(arr, 3)

print(newarr)
# %%
"""Split the 2-D array into three 2-D arrays along rows."""
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.array_split(arr, 3, axis=1)

print(newarr)
# %%
"""^Same as above: Use the hsplit() method to split the 2-D array into three 2-D arrays along rows."""
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.hsplit(arr, 3)

print(newarr)
# %%
"""Search"""
"""Find the indexes where the value is 4:"""
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 4, 4])

x = np.where(arr == 4)

print(x)
# %%
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

x = np.where(arr%2 == 0)

print(arr[x])
# %%
"""Find the indexes where the value 7 should be inserted:"""
"""Array should already be SORTED"""
import numpy as np

arr = np.array([6, 7, 8, 9])

x = np.searchsorted(arr, 7)

print(x)
# %%
"""Sort the array: returns new array"""
import numpy as np

arr = np.array([3, 2, 0, 1])

print(np.sort(arr))
# %%
"""Sort a 2-D array:"""
import numpy as np

arr = np.array([[3, 2, 4], [5, 0, 1]])

print(np.sort(arr))
# %%
"""Filtering Arrays: In NumPy, you filter an array using a boolean index list."""
import numpy as np

arr = np.array([41, 42, 43, 44])

x = [True, False, True, False]
# print(x)
newarr = arr[x]

print(newarr)
# %%
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

filter_arr = arr % 2 == 0
filter_arr = [i%2!=0 for i in arr] #same as above

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)