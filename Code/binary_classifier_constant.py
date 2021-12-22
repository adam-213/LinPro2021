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

significant_digits = 6
epsilon = 10**(-significant_digits)


a_x = np.vstack((X, -np.ones(N)))
a_x = -a_x.transpose()

a_y = np.vstack((Y, -np.ones(N)))
a_y = a_y.transpose()

a = np.vstack((a_x, a_y))

b = -np.ones(M + N)

c = np.zeros(n + 1)

m = ["highs-ds", "highs-ipm", "revised simplex", "interior-point"]

for method in m[2:]:
    result = linprog(c, A_ub=a, b_ub=b, method=method, bounds=(None, None))
    print(result)
    print(method)

    result = result.x
    result[abs(result) < epsilon] = 0

    a_result = result[:-1]
    b_result = result[-1]

    print("a:")
    print(a_result)
    print("b: ", b_result)

    ## Chcem sa uistit lebo neverim

    DEBUG_PRINT_ALL = False
    
    for x in X.transpose():
        if DEBUG_PRINT_ALL:
            print((a_result @ x) - b_result)
        if not (round((a_result @ x) - b_result, significant_digits) >= 1):
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
