import numpy as np
from petsc4py import PETSc
from Ncyl.tds import scattering, cyl
from utils import example_setups
print(PETSc.ScalarType)
N = 10
cyls =  example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3)
sc = scattering(cyls, freq=10, use_2cyl_modes=False)
A = sc.make_scattering_blocks()
b = sc.make_rhs_vector()

#convert to petsc types
Ap = PETSc.Mat().createDense(A.shape, array=A)
bp = PETSc.Vec().createWithArray(b)

#set up gmres
ksp = PETSc.KSP().create()

#no preconditioner
pc = ksp.getPC()
pc.setType('none')

ksp.setOperators(Ap)
ksp.setFromOptions()
x0 = PETSc.Vec().createWithArray(np.zeros(len(b)))
x = ksp.solve(bp, x0)
print(x)
