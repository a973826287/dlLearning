import numpy as np

def generate_numpy_matrix(m, n):
    a = np.random.randint(10, size = m * n, dtype = np.int32) + 10
    a = a.reshape(m , n)
    return a


matrix = generate_numpy_matrix(10, 5)
print (matrix)
print (matrix[0]) # first row
print (matrix[:, 0]) # first column
print (matrix[0::2]) # odd rows

matrix1 = generate_numpy_matrix(3, 4)
matrix2 = generate_numpy_matrix(4, 5)
print(matrix1)
print(matrix2)
result = np.matmul(matrix1, matrix2)#叉乘
print(result)
#叉乘
result = matrix1 @ matrix2
print(result)

result = matrix1 * 3 + 2
print(result)

matrix3 = generate_numpy_matrix(3, 4)

#点乘
result = matrix1 * matrix3
print (matrix3)
print (result)
print("===================================================")
matrix1 = generate_numpy_matrix(3, 4)
matrix2 = generate_numpy_matrix(3, 4)
print(np.transpose(matrix1))# matrix transpose
print(matrix1.T)#for python3.5 or higher version
print(matrix1)
print(np.sum(matrix1))
#axis represent 第几个纬度，like 0 竖着加， 写的时候试一下
print(np.sum(matrix1, axis = 0))

print(np.vstack([matrix1, matrix2]))# combine two matrix in row
print(np.hstack([matrix1, matrix2]))# combine tow matrix in column
