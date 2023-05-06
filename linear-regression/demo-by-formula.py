from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

# height (cm)
"""
[[147]
 [150]
 [153]
 [158]
 [163]
 [165]
 [168]
 [170]
 [173]
 [175]
 [178]
 [180]
 [183]]
"""
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
"""
[[49]
 [50]
 [51]
 [54]
 [58]
 [59]
 [60]
 [62]
 [63]
 [64]
 [66]
 [67]
 [68]]
"""
Y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

print(X)
print(Y)

# Visualize data 
plt.plot(X, Y, 'ro')

# # Set axis
xmin = 140
xmax = 190
ymin = 45
ymax = 75
plt.axis([xmin, xmax, ymin, ymax])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

"""
(cân nặng) = w_1*(chiều cao) + w_0
Tiếp theo, chúng ta sẽ tính toán các hệ số w_1 và w_0 dựa vào công thức (5)
"""

# Building Xbar 
"""
X.shape[0] = 13
(X.shape[0], 1) = (13, 1)
np.ones((X.shape[0], 1)) -> array 13 dòng 1 cột với các phần tử có value = 1
"""
one = np.ones((X.shape[0], 1))
"""
[[1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]]
 """
Xbar = np.concatenate((one, X), axis = 1)
"""
[[  1. 147.]
 [  1. 150.]
 [  1. 153.]
 [  1. 158.]
 [  1. 163.]
 [  1. 165.]
 [  1. 168.]
 [  1. 170.]
 [  1. 173.]
 [  1. 175.]
 [  1. 178.]
 [  1. 180.]
 [  1. 183.]]
 """

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, Y)
"""
Chú ý: giả nghịch đảo của một ma trận A trong Python sẽ được tính bằng numpy.linalg.pinv(A)
Với pinv là từ viết tắt của pseudo inverse.
"""
w = np.dot(np.linalg.pinv(A), b)

print('w = ', w)

# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
print(x0)
y0 = w_0 + w_1*x0

# Drawing the fitting line
# data
plt.plot(X.T, Y.T, 'ro')
# the fitting line
plt.plot(x0, y0)

plt.axis([xmin, xmax, ymin, ymax])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

"""
Từ đồ thị bên trên ta thấy rằng các điểm dữ liệu màu đỏ nằm khá gần đường thẳng dự đoán màu xanh. 
Vậy mô hình Linear Regression hoạt động tốt với tập dữ liệu training. 
Bây giờ, chúng ta sử dụng mô hình này để dự đoán cân nặng của hai người 
có chiều cao 155 và 160 cm mà chúng ta đã không dùng khi tính toán nghiệm.
"""
y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )