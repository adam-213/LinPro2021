from scipy.io import loadmat

vars = loadmat('sp_ln_sp_data.mat')
n = vars['n']
N = vars['N']
M = vars['M']
X = vars['X']
Y = vars['Y']