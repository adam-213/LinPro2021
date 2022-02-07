import pandas as pd
import numpy as np
from tqdm import tqdm
import sklearn.model_selection as skm
from scipy.optimize import linprog
from scipy.stats import loguniform

data = pd.read_csv("data_banknote_authentication (1).txt")
cNames = ["variance", "skewness", "curtosis", "entropy", "isForgery"]
data.set_axis(cNames, axis=1, inplace=True)
train, test = skm.train_test_split(data)
significant_digits = 6
epsilon = 10 ** (-significant_digits)

m = ["highs-ds", "highs-ipm", "revised simplex", "interior-point"]

train_no = train.loc[train["isForgery"] == 1].iloc[:, [0, 1, 2, 3]]
train_yes = train.loc[train["isForgery"] == 0].iloc[:, [0, 1, 2, 3]]
# split to X,Y by is forgery and cut the column out

X = train_yes.to_numpy()
Y = train_no.to_numpy()

# X plna sirka
Xp = np.hstack((-X, np.ones(X.shape[0])[:, None]))
Xp = np.hstack((Xp, np.zeros([X.shape[0], X.shape[1]])))
Xp = np.hstack((Xp, -np.eye(X.shape[0])))
Xp = np.hstack((Xp, np.zeros([X.shape[0], Y.shape[0]])))
# Y plna sirka
Yp = np.hstack((Y, -np.ones(Y.shape[0])[:, None]))
Yp = np.hstack((Yp, np.zeros([Y.shape[0], Y.shape[1]])))
Yp = np.hstack((Yp, np.zeros([Y.shape[0], X.shape[0]])))
Yp = np.hstack((Yp, -np.eye(Y.shape[0])))
# spodne 2 riadky matice
m1 = np.hstack((np.eye(X.shape[1]), np.zeros(X.shape[1])[:, None]))
m2 = np.hstack((m1, -np.eye(X.shape[1])))
m3 = np.hstack((m2, np.zeros([X.shape[1], X.shape[0]])))
m4 = np.hstack((m3, np.zeros([Y.shape[1], Y.shape[0]])))

n1 = np.hstack((-np.eye(X.shape[1]), np.zeros(X.shape[1])[:, None]))
n2 = np.hstack((n1, -np.eye(X.shape[1])))
n3 = np.hstack((n2, np.zeros([X.shape[1], X.shape[0]])))
n4 = np.hstack((n3, np.zeros([Y.shape[1], Y.shape[0]])))

# vsetky riadky stacknute na seba
A = np.vstack((Xp, Yp))
A = np.vstack((A, m4))
A = np.vstack((A, n4))

b = np.hstack((-np.ones(X.shape[0] + Y.shape[0]), np.array([0] * 8)))



rvs = loguniform.rvs(1e-2, 1e4, size=1000)

options = {"cholesky": False, "lstsq": True, "presolve": True, "sym_pos": False}

uvSuma = []
mu = []
kardinalita = []
acc = []

for mi in tqdm(rvs):
    Xcor = 0
    Ycor = 0
    icor = 0
    c = np.array([0] * (X.shape[1] + 1) + [mi] * X.shape[1] + [1] * X.shape[0] + [1] * Y.shape[0])
    result = linprog(c, A_ub=A, b_ub=b, method="highs-ds", bounds=((0, None)))
    try:
        result = result.x
        mu.append(mi)
        kardinalita.append((4 - list(result[0:4]).count(0)))
        uvSuma.append(sum(result[9:]))
        for row in test.to_numpy():
            if sum(result[0:4] * row[:-1]) - result[5] > -1 and row[-1] == 0:
                Xcor += 1
            elif sum(result[0:4] * row[:-1]) - result[5] < -1 and row[-1] == 1:
                Ycor += 1
            else:
                icor += 1
        acc.append((Xcor + Ycor) / len(test.to_numpy()))
    except Exception:
        pass

import matplotlib.pyplot as plt
import seaborn as sns
sb = sns.regplot(mu,acc,line_kws={"color":"blue"})
sb.set_xlabel("Parameter µ")
sb.set_ylabel("Presnosť klasifikácie")
plt.show()
#
# sb1 = sns.regplot(kardinalita,uvSuma,line_kws={"color":"blue"})
# sb1.set_ylabel("Ф(µ)")
# sb1.set_xlabel("Kardinalita a")
# plt.show()
#
# sb2 = sns.regplot(kardinalita,acc,line_kws={"color":"blue"})
# sb2.set_xlabel("Kardinalita a")
# sb2.set_ylabel("Presnosť klasifikácie")
# plt.show()

# sb3 = sns.regplot(mu,uvSuma,line_kws={"color":"blue"})
# sb3.set_xlabel("Parameter µ")
# sb3.set_ylabel("Ф(µ)")
# plt.show()
