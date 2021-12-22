from scipy.io import loadmat
from pathlib import Path
root = Path(__file__).parent.parent

def loadvars():
    vars = loadmat(root.as_posix() +"/"+ "Data" +"/" + 'sp_ln_sp_data.mat')
    n = vars['n']
    N = vars['N']
    M = vars['M']
    X = vars['X']
    Y = vars['Y']
    
    
loadvars()