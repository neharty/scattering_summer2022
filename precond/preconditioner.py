import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals, block_diag, qr
from scipy.sparse.linalg import svds

class preconditioner():
    '''
        preconditioner template class for
        scattering-based preconditioners
    '''
    def __init__(self, sc):
        self.sc = sc
        self.inv_mat = None

class hbs(preconditioner):
    def __init__(self, sc, tol = 1e-10, level = 1):
        super().__init__(sc)
        self.mat_block = self.sc.make_scattering_blocks_blockform()
        self.dist_mat = sc.dist_matrix
        self.dist = sc.precond_dist
        self.tol = tol
        self.level = level
        self.inv_mat = self.construct_Ainv(self.mat_block, self.tol, self.level)

    def construct_Aapprox_blockform(self, A_b, tol):
        #A_b is the block form of A, as a list 
        epsrank = np.zeros((len(A_b), len(A_b[0])))
        for i in range(len(A_b)):
            for j in range(len(A_b[0])):
                svdvect = svdvals(np.array(A_b[i][j]))
                if i != j:
                    #if self.dist_mat[i, j] >= self.dist:
                    epsrank[i, j] = len(svdvect[svdvect > tol])
                else:
                    epsrank[i, j] = 0

        #print(epsrank)

        rows, cols = len(A_b), len(A_b[0])

        U = np.array([None for n in range(rows)])
        Y = np.array([None for n in range(rows)])
        V = np.array([None for n in range(cols)])
        for i in range(rows):
            colarr = np.arange(cols)
            colarr = np.delete(colarr, i)
            B = [A_b[i][c] for c in colarr]
            B = np.block(B)
            #B = np.concatenate(np.array(A_b[i][colarr]), axis=1)
            U_i, X_i, Y_i = svd(B)#, k=epsrank)
            Y_i = Y_i.conj().T
            U[i] = U_i[:, :int(np.max(epsrank[i, :]))]
            Y[i] = Y_i[:, :int(np.max(epsrank[i, :]))]
            print(B.shape, U_i.shape, X_i.shape, Y_i.shape)

        A_bH = [[None for i in range(rows)] for j in range(cols)]

        for i in range(cols):
            for j in range(rows):
                A_bH[i][j] = A_b[j][i].conj().T
        
        for i in range(cols):
            rowarr = np.arange(rows)
            rowarr = np.delete(rowarr, i)
            B = [A_bH[i][r] for r in rowarr]
            B = np.block(B)
            V_i, X_i, Y_i = svd(B)#, k=epsrank)
            V[i] = V_i[:, :int(np.max(epsrank[:, i]))]
            #print(U[i] @ V[i].conj().T)

        Atilde = [[None for i in range(cols)] for j in range(rows)]
        D = [None for i in range(cols)]
        for i in range(rows):
            for j in range(cols):
                if i != j:
                    Atilde[i][j] = U[i].conj().T @ A_b[i][j] @ V[j]
                else:
                    n, no = U[i].conj().T.shape
                    mo, m = V[j].shape
                    Atilde[i][j] = np.zeros((n, m))
                    D[i] = A_b[i][j]
        #print(Atilde.shape)
        return U, Atilde, V, D

    def construct_Aapprox_svd(self, A_b, tol):
        U, Atilde, V, D = self.construct_Aapprox_blockform(A_b, tol)
        Atilde = np.block(Atilde)
        U = block_diag(*U)
        V = block_diag(*V)
        D = block_diag(*D)
        return U, Atilde, V, D

    def construct_Ainv_matrices(self, U, V, D):
        Dinv = np.linalg.inv(D)
        Dhat = np.linalg.inv(V.conj().T @ Dinv @ U)
        E = Dinv @ U @ Dhat
        F = (Dhat @ V.conj().T @ Dinv).conj().T
        G = Dinv - (Dinv @ U @ Dhat @ V.conj().T @ Dinv)
        return E, F, Dhat, G

    def construct_Ainv(self, A_b, tol, l):
        #print(l)
        U, Atilde, V, D = self.construct_Aapprox_blockform(A_b, tol)
        
        U = block_diag(*U)
        V = block_diag(*V)
        D = block_diag(*D)

        E, F, Dhat, G = self.construct_Ainv_matrices(U, V, D)

        if l == 1:
            Atilde = np.block(Atilde)
            return E @ np.linalg.inv(Atilde + Dhat) @ F.conj().T + G

        Atmp = np.array(self.reblock_blockform(Atilde))
        print(Atmp.shape)
        rows, cols, a, b = Atmp.shape
        
        #make Dhat a block matrix
        Dhatblock = [[None for i in range(cols)] for j in range(rows)]
        for i in range(rows):
            for j in range(cols):
                Dhatblock[i][j] = Dhat[i*a:(i+1)*a, j*b:(j+1)*b].tolist()

        Dhatblock = np.array(Dhatblock)

        return E @ self.construct_Ainv(Atmp + Dhatblock, tol, l-1) @ F.conj().T + G

    def reblock_blockform(self, A_b):
        #works if A_b has an even number of blocks
        rows, cols, a, b = A_b.shape
        rows = int(rows/2)
        cols = int(cols/2)
        A_new = [[None for i in range(cols)] for j in range(rows)]
        for i in range(rows):
            for j in range(cols):
                block = [[A_b[2*i, 2*j], A_b[2*i, 2*j + 1]], [A_b[2*i + 1, 2*j], A_b[2*i + 1, 2*j + 1]]]
                A_new[i][j] = np.block(block)

        return A_new #return as list

class antoine(preconditioner):
    def __init__(self, sc):
        super().__init__(sc)
        self.mat_block = self.sc.make_scattering_blocks_blockform()
        self.dist = sc.precond_dist
        self.dist_mat = sc.dist_matrix
        self.inv_mat = self.make_precond_matrix()

    def make_precond_matrix(self):
        dims = len(self.mat_block), len(self.mat_block[0])
        for i in range(dims[0]):
            for j in range(dims[1]):
                if i == j:
                    pass
                else:
                    if self.dist_mat[i, j] > self.dist:
                        rows, cols = np.array(self.mat_block[i][j]).shape
                        self.mat_block[i][j] = np.zeros((rows, cols))
        self.mat = np.block(self.mat_block)
        matdim = self.mat.shape[0]
        return 2*np.eye(matdim) - self.mat

class neumann_lowrank(preconditioner):
    def __init__(self, sc, tol = 1e-8, neumann_terms=2):
        super().__init__(sc)
        self.mat_block = self.sc.make_scattering_blocks_blockform()
        self.dist_mat = self.sc.dist_matrix
        self.dist = self.sc.precond_dist
        self.tol = tol
        self.neumann_terms = neumann_terms
        self.inv_mat = self.make_precond_matrix()

    def low_rank_factorization(self, blk):
        #returns low rank factorization of a matrix, given a tolerance, using the SVD
        #TODO upgrade to QR factorization
        '''
        epsrank = np.argmin(np.abs(svdvals(blk) - self.tol)) + 1
        U, S, Vh = svd(blk)
        return U[:, :epsrank] @ np.diag(S[:epsrank]) @ Vh[:epsrank, :]
        '''
        #Randomized QR factorization
        m, n = blk.shape

        #do once for initialization
        R = np.random.normal(size=(n, 1))
        Q, R = qr(blk@R)
        B = np.conjugate(Q.T) @ blk
        blk = blk - Q @ B

        while np.linalg.norm(blk) > self.tol:
            R = np.random.normal(size=(n, l))
            Qnew, R = qr(blk@R)
            Bnew = np.conjugate(Qnew.T) @ blk
            Qtmp = np.vstack((Q.T, Qnew.T))
            Q = Qtmp.T
            B = np.vstack((B, Bnew))
            blk = blk - Qnew @ Bnew

        return Q@B

    def neumann_series(self, A, N):
        B = np.eye(*A.shape, dtype=A.dtype)
        for i in range(1, N):
            B += np.linalg.matrix_power(np.eye(*A.shape, dtype=A.dtype) - A, i)
        return B

    def make_precond_matrix(self):
        dims = len(self.mat_block), len(self.mat_block[0])
        for i in range(dims[0]):
            for j in range(dims[1]):
                if i == j:
                    pass
                else:
                    #if self.dist_mat[i, j] > self.dist:
                    #print('in here!')
                    rows, cols = np.array(self.mat_block[i][j]).shape
                    self.mat_block[i][j] = self.low_rank_factorization(self.mat_block[i][j])

        self.mat = np.block(self.mat_block)
        matdim = self.mat.shape[0]
        #return self.neumann_series(self.mat, self.neumann_terms)
        return 2*np.eye(matdim) - self.mat

