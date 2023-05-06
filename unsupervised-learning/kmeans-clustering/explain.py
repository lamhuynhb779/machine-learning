import numpy as np

# matrix 5 row 2 column
V = np.random.rand(5,2)
print(V)

ALabel = np.asarray([0]*1 + [1]* 3 + [2]*1)
print(ALabel)

print(ALabel == 1) # -> [False  True  True  True False]

B = V[ALabel == 1, :]
print(B)

# It is a notation used in Numpy/Pandas.

# [ : , 0 ] means (more or less) [ first_row:last_row , column_0 ]. 
# If you have a 2-dimensional list/matrix/array, this notation will give you all values in column 0 (from all rows).
print(B[:, 0])
print(B[:, 1])


# markersize là kích thước của marker
# alpha là độ trong suốt của marker
