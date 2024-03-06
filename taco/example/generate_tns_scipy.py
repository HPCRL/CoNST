import sparse
import numpy as np

def generate_3index():
    P = 600
    Q = 600
    R = 600
    #S = 500
    A = 50
    B = 50
    C = 50
    
    start = sparse.random((P, Q, R), density = 0.05)
    with open("A_scipy.tns", "w") as f:
        for ind, d in enumerate(start.data):
            f.write("{} {} {} {}\n".format(start.coords[0][ind] + 1, start.coords[1][ind] + 1, start.coords[2][ind] + 1, d))
    
    c2 = sparse.random((C, R), density = 0.05)
    with open("C2_scipy.tns", "w+") as gfile:
        for ind, d in enumerate(c2.data):
            gfile.write("{} {} {}\n".format(c2.coords[0][ind] + 1, c2.coords[1][ind] + 1, d))
    
    c3 = sparse.random((B, Q), density = 0.05)
    with open("C3_scipy.tns", "w+") as gfile:
        for ind, d in enumerate(c3.data):
            gfile.write("{} {} {}\n".format(c3.coords[0][ind] + 1, c3.coords[1][ind] + 1, d))
    
    c4 = sparse.random((A, P), density = 0.05)
    with open("C4_scipy.tns", "w+") as gfile:
        for ind, d in enumerate(c4.data):
            gfile.write("{} {} {}\n".format(c4.coords[0][ind] + 1, c4.coords[1][ind] + 1, d))

def generate_ttmc_dense_matrices():
    #I = 183
    J = 24
    K = 1140
    L = 1717

    A = 50
    B = 50
    C = 50

    m1 = np.random.rand(J, A)
    with open("m1.tns", "w+") as gfile:
        for x in range(J):
            for y in range(A):
                gfile.write("{} {} {}\n".format(x+1, y+1, m1[x][y]))
    
    m2 = np.random.rand(K, B)
    with open("m2.tns", "w+") as gfile:
        for x in range(K):
            for y in range(B):
                gfile.write("{} {} {}\n".format(x+1, y+1, m2[x][y]))   

    m3 = np.random.rand(L, C)
    with open("m3.tns", "w+") as gfile:
        for x in range(L):
            for y in range(C):
                gfile.write("{} {} {}\n".format(x+1, y+1, m3[x][y]))

def generate_3cent_int():
    MO = 10
    PAO = 20
    PNO = 30
    AUX = 50
    teoo = sparse.random((MO, MO, AUX), density = 0.03)
    with open("teoo.tns", "w") as f:
        for ind, data in enumerate(teoo.data):
            f.write("{} {} {} {}\n".format(teoo.coords[0][ind] + 1, teoo.coords[1][ind] + 1, teoo.coords[2][ind] + 1, data))
    teov = sparse.random((MO, PAO, AUX), density = 0.03)
    with open("teov.tns", "w") as f:
        for ind, data in enumerate(teov.data):
            f.write("{} {} {} {}\n".format(teov.coords[0][ind] + 1, teov.coords[1][ind] + 1, teov.coords[2][ind] + 1, data))
    d = sparse.random((MO, PAO, PNO), density = 0.03)
    with open("d.tns", "w") as f:
        for ind, data in enumerate(d.data):
            f.write("{} {} {} {}\n".format(d.coords[0][ind] + 1, d.coords[1][ind] + 1, d.coords[2][ind] + 1, data))

def generate_dlpno():
    MO = 200
    AUX = 500
    PNO = 50
    PAO = 800
    teov = sparse.random((MO, PAO, AUX), density = 0.03)
    d = sparse.random((MO, PAO, PNO), density = 0.03)
    with open("teov_scipy.tns", "w") as f:
        for ind, data in enumerate(teov.data):
            f.write("{} {} {} {}\n".format(teov.coords[0][ind] + 1, teov.coords[1][ind] + 1, teov.coords[2][ind] + 1, data))
    with open("d_scipy.tns", "w") as f:
        for ind, data in enumerate(d.data):
            f.write("{} {} {} {}\n".format(d.coords[0][ind] + 1, d.coords[1][ind] + 1, d.coords[2][ind] + 1, data))
generate_3cent_int()
