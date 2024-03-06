import numpy as np
from scipy.io import mmread, mmwrite


mat1 = mmread("bcsstk17.mtx")
mat2 = mmread("bcsstk17.mtx")
mat_res = mat1 @ mat2
mmwrite("res_dense.mtx", mat_res)
