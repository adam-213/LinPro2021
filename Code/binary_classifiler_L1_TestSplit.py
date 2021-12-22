import numpy as np
from scipy.optimize import linprog
from scipy.io import loadmat


from pathlib import Path
root = Path(__file__).absolute().parent.parent
path = root.as_posix() + "/" + "Data" + "/" + 'sp_ln_sp_data.mat'
vars = loadmat(path)
n = vars['n'][0][0]
N = vars['N'][0][0]
M = vars['M'][0][0]
X = vars['X']
Y = vars['Y']

epsilon = 10 ** (-6)

testx = X[:, 45:]
X = X[:, 0:45]
testy = Y[:, 45:]
Y = Y[:, 0:45]
N = M = 45

# inequalities for x, where
# -(x^T)a - b <= -1
# we add a row (column after transpose) of ones because a and b 
# will be in the same vector (a_1, ..., a_n, b)
a_x = np.vstack((X, -np.ones(N)))
a_x = -a_x.transpose()

# inequalities for y, where
# (y^T)a - b <= -1
a_y = np.vstack((Y, -np.ones(M)))
a_y = a_y.transpose()

# inequalities for x and y stacked
a = np.vstack((a_x, a_y))

# add 0s to the right
a = np.hstack((a, np.zeros((M + N, n))))

# add extra inequalities we use for the absolute value
for x in 1, -1:
    inequality = np.hstack((x * np.eye(n), np.zeros((n, 1)), -np.eye(n)))
    a = np.vstack((a, inequality))



# right side for inequalities, -1 for x and y and 0 when were making the abs work
b = -np.ones(M + N)
b = np.concatenate((b, np.zeros(2 * n)))

# result.x is in the form [a_1, ..., a_n, b, t_1, ..., t_n]
# and after the abs val transformation we're minimizing (1^T)t
c = np.zeros(n + 1)
c = np.concatenate((c, np.ones(n)))
print(a.shape, b.shape)
m = ["highs-ds", "highs-ipm", "revised simplex", "interior-point"]
for method in m:
    result = linprog(c, A_ub=a, b_ub=b, method=method, bounds=(None, None))

    result = result.x
    result[abs(result) < epsilon] = 0

    a_result = result[:n]
    b_result = result[n]

    print("a:")
    print(a_result)
    print("b:", b_result)

    print("should be x aka >0")
    for row in testx.T:
        print(a_result.T.dot(row)+ b_result >0)
    
    print("should be y aka <0")
    for row in testy.T:
        print(a_result.T.dot(row) + b_result <0)
