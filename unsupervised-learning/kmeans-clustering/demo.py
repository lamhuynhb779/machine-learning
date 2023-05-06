from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# Chúng ta cũng cần thêm thư viện scipy.spatial.distance để tính khoảng cách 
# giữa các cặp điểm trong hai tập hợp một cách hiệu quả
from scipy.spatial.distance import cdist

np.random.seed(11)

# lấy các điểm theo phân phối chuẩn có kỳ vọng tại các điểm có tọa độ (2, 2), (8, 3) và (3, 6)
means = [[2, 2], [8, 3], [3, 6]]

# ma trận hiệp phương sai giống nhau và là ma trận đơn vị
coveriance = [[1, 0], [0, 1]]

# Mỗi cluster có 500 điểm
N = 500

# Chú ý rằng mỗi điểm dữ liệu là một hàng của ma trận dữ liệu.
X0 = np.random.multivariate_normal(means[0], coveriance, N)
X1 = np.random.multivariate_normal(means[1], coveriance, N)
X2 = np.random.multivariate_normal(means[2], coveriance, N)

X = np.concatenate((X0, X1, X2), axis = 0)

# Chia làm 3 clusters
K = 3

# Example np.asarray([0]*1 + [1]*2 + [2]*3) = [0 1 1 2 2 2]
original_label = np.asarray([0]*N + [1]*N + [2]*N)

print(original_label)

from func import kmeans_display


kmeans_display(X, original_label)
