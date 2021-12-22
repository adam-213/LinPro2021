import numpy as np
from scipy.optimize import linprog
from scipy.io import loadmat
import Load_Data
epsilon = 10 ** (-6)

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

b = np.empty(100)
b.fill(-1)
print(b)

c = np.zeros(n + 1)

m = ["highs-ds", "highs-ipm", "highs", "revised simplex", "interior-point", "simplex"]
for u in m:
    print(u)
    result = linprog(c, A_ub=a, b_ub=b, method=u, bounds=(None, None))


    result = result.x

    a_result = result[:50]
    b_result = result[50]
    with open(u +".txt","w") as c:
        c.write(f"a: {a_result}")
        c.write("\n")
        c.write(f"b: {b_result}")

    # for i, v in enumerate(a):
    #     out = result.x @ v
    #     print(i, "|", out if abs(out) > epsilon else 0)

    # exit()

    # Chcem sa uistit lebo neverim

    x_fucked = False
    for x in X.transpose():
        # print(a_result @ x - b_result)

        # print(round(a_result @ x - b_result, 8))
        if not (round(a_result @ x - b_result, 8) >= 1):
            print("X fucked up")
            x_fucked = True
            # break
    else:
        print("X all good")

    for y in Y.transpose():
        # print((a_result @ y) - b_result)
        # print(round((a_result @ y) - b_result, 8))
        if not (round((a_result @ y) - b_result, 8) <= -1):
            print("Y fucked up")
            break
    else:
        print("Y all good")

# signifinatne_indexy = [i for i, a in enumerate(a_result) if a]
# print(signifinatne_indexy)
