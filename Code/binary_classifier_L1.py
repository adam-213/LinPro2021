import numpy as np
from scipy.optimize import linprog
from scipy.io import loadmat


from pathlib import Path
root = Path(__file__).absolute().parent.parent
path = root.as_posix() + "/" + "Data" + "/" + 'sp_ln_sp_data.mat'
vars = loadmat(path)
n = vars['n'][0][0] # pocet suradnic vektorov
N = vars['N'][0][0] # pocet vektorov X
M = vars['M'][0][0] # pocet vektorov Y
X = vars['X']
Y = vars['Y']

significant_digits = 6
epsilon = 10**(-significant_digits)


# inequalities for x, where
# -(x^T)a - b <= -1
# we add a row (column after transpose) of ones because a and b 
# will be in the same vector (a_1, ..., a_n, b)
a_x = np.vstack((X, -np.ones(N)))
a_x = -a_x.transpose()

# inequalities for y, where
# (y^T)a - b <= -1
a_y = np.vstack((Y, -np.ones(N)))
a_y = a_y.transpose()

# inequalities for x and y stacked
a = np.vstack((a_x, a_y))

# add 0s to the right
a = np.hstack((a, np.zeros((N + M, n))))

# add extra inequalities we use for the absolute value
for x in 1, -1:
    inequality = np.hstack((x*np.eye(n), np.zeros((n, 1)), -np.eye(n)))
    a = np.vstack((a, inequality))



# right side for inequalities, -1 for x and y and 0 when were making the abs work
b = -np.ones(M + N)
b = np.concatenate((b, np.zeros(2*n)))

# result.x is in the form [a_1, ..., a_n, b, t_1, ..., t_n]
# and after the abs val transformation we're minimizing (1^T)t
c = np.zeros(n + 1)
c = np.concatenate((c, np.ones(n)))

m = ["highs-ds", "highs-ipm", "revised simplex", "interior-point"]

for method in m[2:]:
    result = linprog(c, A_ub=a, b_ub=b, method=method, bounds=(None, None))
    print(result)
    print("Method:", method)

    result = result.x
    result[abs(result) < epsilon] = 0

    a_result = result[:n]
    b_result = result[n]

    print("a:")
    print(a_result)
    print("b:", b_result)


    ## Chcem sa uistit lebo neverim

    DEBUG_PRINT_ALL = False

    for x in X.transpose():
        if DEBUG_PRINT_ALL:
            print(a_result @ x - b_result)
        if not (round(a_result @ x - b_result, significant_digits) >= 1):
            print("X fucked up")
            break
    else:
        print("X all good")

    for y in Y.transpose():
        if DEBUG_PRINT_ALL:
            print((a_result @ y) - b_result)
        if not (round((a_result @ y) - b_result, significant_digits) <= -1):
            print("Y fucked up")
            break
    else:
        print("Y all good")


    signifikatne_indexy = [i for i, a in enumerate(a_result) if a]
    print("Signifikantne indexy:", signifikatne_indexy)
    print(f"Pocet signifikantnych atribut: {len(signifikatne_indexy)} / {n}")
    print("-"*160)
