import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals, block_diag, qr, diagsvd
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
    def __init__(self, sc, tol = 1e-8, level = 1, method='new'):
        super().__init__(sc)
        self.mat_block = self.sc.make_scattering_blocks_blockform()
        self.dist_mat = sc.dist_matrix
        self.dist = sc.precond_dist
        self.tol = tol
        self.level = level
        self.method = method
        self.inv_mat = self.construct_Ainv(self.mat_block, self.level)
        
    def construct_Aapprox_blockform(self, A_b):
        #A_b is the block form of A, as a list 
        epsrank = np.zeros((len(A_b), len(A_b[0])))
        for i in range(len(A_b)):
            for j in range(len(A_b[0])):
                svdvect = svdvals(np.array(A_b[i][j]))
                if i != j:
                    #if self.dist_mat[i, j] >= self.dist:
                    epsrank[i, j] = len(svdvect[svdvect > self.tol])
                else:
                    epsrank[i, j] = 0
        epsrank = int(np.amax(epsrank))
        #print(epsrank)

        rows, cols = len(A_b), len(A_b[0])
        
        U = [None for n in range(rows)]
        AVblk = [None for n in range(rows)]
        Atilde = [[None for i in range(cols)] for j in range(rows)]
        V = [None for n in range(cols)]

        for i in range(rows):
            colarr = np.arange(cols)
            colarr = np.delete(colarr, i)
            B = [A_b[i][c] for c in colarr]
            B = np.block(B)
            m, n = B.shape
            #B = np.concatenate(np.array(A_b[i][colarr]), axis=1)
            U_i, X_i, Y_i = svd(B)#, k=epsrank)
            U[i] = U_i[:, :epsrank]
            
            X_i = diagsvd(X_i, m, n)
            tmpmat = X_i @ Y_i
            AVblk[i] = (X_i @ Y_i)
    
        if self.method == 'new':
            #print('new method')
            AVblk = np.vstack(AVblk)

            for j in range(cols):
                rowarr = np.delete(np.arange(rows), j)
                rlen, clen = A_b[j][j].shape
                #compute V[j] via an SVD
                if j == 0:
                    #compute V[0]
                    tmpmat = AVblk[rlen:, :clen]
                    m, n = tmpmat.shape
                    Utmp, Stmp, Vtmp = svd(tmpmat)
                    V[0] = Vtmp.conj().T[:, :epsrank]
                elif j == cols-1:
                    #compute V[-1] 
                    tmpmat = AVblk[:j*rlen, (j-1)*clen:]
                    m, n = tmpmat.shape
                    Utmp, Stmp, Vtmp = svd(tmpmat)
                    V[-1] = Vtmp.conj().T[:, :epsrank]
                else:
                    #compute V[j]
                    #note, this assumes all blocks are the same size
                    #print(j)
                    tmpmat1 = AVblk[:(j-1)*rlen, (j-1)*clen:j*clen]
                    tmpmat2 = AVblk[(j+1)*rlen:, (j-1)*clen:j*clen]
                    tmpmat = np.vstack((tmpmat1, tmpmat2))
                    m, n = tmpmat.shape
                    Utmp, Stmp, Vtmp = svd(tmpmat)
                    Atildecol = Utmp @ diagsvd(Stmp, m, n)
                    V[j] = Vtmp.conj().T[:, :epsrank]
        
        if self.method == 'old':
            print('old method')
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
                V[i] = V_i[:, :epsrank]
                #print(U[i] @ V[i].conj().T)
        
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

        return U, Atilde, V, D
    
    def construct_Ainv_matrices(self, U, V, D):
        Dinv = np.linalg.inv(D)
        #plt.show()
        Dhat = np.linalg.inv(V.conj().T @ Dinv @ U)
        #Dhat = U.conj().T @ D @ V
        E = Dinv @ U @ Dhat
        F = Dhat @ V.conj().T @ Dinv
        G = Dinv - (Dinv @ U @ Dhat @ V.conj().T @ Dinv)
        return E, F, Dhat, G

    def construct_Ainv(self, A_b, l):
        #print(l)
        U, Atilde, V, D = self.construct_Aapprox_blockform(A_b)
        
        U = block_diag(*U)
        V = block_diag(*V)
        D = block_diag(*D)

        E, F, Dhat, G = self.construct_Ainv_matrices(U, V, D)

        if l == 1:
            Atilde = np.block(Atilde)
            return E @ np.linalg.inv(Atilde + Dhat) @ F + G
        
        #recursive step
        Atmp = self.reblock_blockform(Atilde)
        #print(Atmp.shape)
        rows = len(Atmp)
        cols = len(Atmp[0])
        a, b = Atmp[0][0].shape
        
        #make Dhat a block matrix
        Dhatblock = [[None for i in range(cols)] for j in range(rows)]
        for i in range(rows):
            for j in range(cols):
                Dhatblock[i][j] = Dhat[i*a:(i+1)*a, j*b:(j+1)*b].tolist()
        
        blkmat = [[None for i in range(cols)] for j in range(rows)]
        for i in range(rows):
            for j in range(cols):
                blkmat[i][j] = Atmp[i][j] + Dhatblock[i][j]

        return E @ self.construct_Ainv(blkmat, l-1) @ F + G

    def reblock_blockform(self, A_b):
        #works if A_b has an even number of blocks
        rows = len(A_b)
        cols = len(A_b[0])
        #rows, cols, a, b = A_b.shape
        rows = int(rows/2)
        cols = int(cols/2)
        A_new = [[None for i in range(cols)] for j in range(rows)]
        for i in range(rows):
            for j in range(cols):
                block = [[A_b[2*i][2*j], A_b[2*i][2*j + 1]], 
                         [A_b[2*i + 1][2*j], A_b[2*i + 1][2*j + 1]]]
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
    def __init__(self, sc, method='svd', tol = 1e-8, neumann_terms=2):
        super().__init__(sc)
        self.mat_block = self.sc.make_scattering_blocks_blockform()
        self.dist_mat = self.sc.dist_matrix
        self.dist = self.sc.precond_dist
        self.tol = tol
        self.neumann_terms = neumann_terms
        self.method = method
        self.inv_mat = self.make_precond_matrix()

    def low_rank_factorization(self, blk):
        #returns low rank factorization of a matrix, given a tolerance
        if self.method == 'svd':
            epsrank = np.argmin(np.abs(svdvals(blk) - self.tol)) + 1
            U, S, Vh = svd(blk)
            return U[:, :epsrank] @ np.diag(S[:epsrank]) @ Vh[:epsrank, :]
        
        #Randomized QR factorization
        if self.method=='qr':
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

class ifmm(preconditioner):
    pass
