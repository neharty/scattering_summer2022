import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from Ncyl.tds import cyl, scattering
from utils import example_setups

#graph the function
err = [1e-4, 1e-6, 1e-8, 1e-10]
for eps in err:
    nfn = lambda x: np.floor(np.log(eps/np.pi)/np.log(x/2) +1)
    ka = np.logspace(-5, 0, num=100)
    plt.semilogx(ka, nfn(ka))
#plt.show()
plt.clf()

#use the formula for 1 cylinder
err = 1e-10
nfn = lambda x: np.floor(((np.log(err*np.sqrt(2/np.pi)) - 1/2)/(np.log(x/2))) + 1)

x = np.append([0, 1.5, 4.5], np.linspace(7, 15, num = 3))
X, Y = np.meshgrid(x, [0])

N = 2

freqs = [
    0.000001,
    0.000005,
    0.00001,
    0.00005,
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05
    ]

abserr = np.zeros((len(freqs), len(x)))
conexact = np.zeros(len(freqs))
conform = np.zeros(len(freqs))
svals = []
#test on a range of x values
for i in range(len(freqs)):
    cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3)
    #cyls = [cyl('d', [0, 0], 1)]
    cylcp = [deepcopy(c) for c in cyls]

    modes = int(nfn(2*np.pi*freqs[i]))

    sc = scattering(
        cyls,
        freq=freqs[i],
        sum_index=modes+3,
        method='explicit'
        )

    print(modes)

    scf = scattering(
        cylcp,
        freq=freqs[i],
        sum_index=modes,
        method='explicit'
        )

    conexact[i] = np.linalg.cond(sc.make_scattering_blocks())
    conform[i] = np.linalg.cond(scf.make_scattering_blocks())

    exact = sc.make_u_sc(X, Y)[0]
    formula = scf.make_u_sc(X, Y)[0]
    print(exact)
    print(formula)
    abserr[i, :] = np.abs(exact - formula)

for j in range(len(x)):
    plt.loglog(freqs, abserr[:, j], '.-', label='x='+str(x[j]))
plt.legend()
plt.xlabel('freq')
plt.ylabel('|u_formula - u_exact|')
#plt.show()
plt.savefig('figs/freq/lowfreq_error.png')
plt.clf()


plt.loglog(freqs, conexact, '.-', label='exact')
plt.loglog(freqs, conform, '.-', label='formula')
plt.legend()
plt.xlabel('freq')
plt.ylabel('scattering matrix cond num')
plt.savefig('figs/freq/lowfreq_condnum.png')
plt.clf()


#repeat the experiment for 'high' frequencies
freqs = [1, 5, 10, 50, 100]
abserr = np.zeros((len(freqs), len(x)))
conexact = np.zeros(len(freqs))
conform = np.zeros(len(freqs))
svals = []
#test on a range of x values
for i in range(len(freqs)):
    cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 3)
    cylcp = [deepcopy(c) for c in cyls]

    scf = scattering(
        cylcp,
        freq=freqs[i],
        )

    sc = scattering(
        cyls,
        freq=freqs[i],
        method='explicit'
        )

    conexact[i] = np.linalg.cond(sc.make_scattering_blocks())
    conform[i] = np.linalg.cond(scf.make_scattering_blocks())

    print('computed condition numbers')
    
    exact = sc.make_u_sc(X, Y)[0]
    formula = scf.make_u_sc(X, Y)[0]
    
    print('computed x-values')

    print(exact)
    print(formula)
    abserr[i, :] = np.abs(exact - formula)

for j in range(len(x)):
    plt.loglog(freqs, abserr[:, j], '.-', label='x='+str(x[j]))
plt.legend()
plt.xlabel('freq')
plt.ylabel('|u_formula - u_exact|')
plt.savefig('figs/freq/highfreq_error.png')
plt.clf()

plt.loglog(freqs, conexact, '.-', label='exact')
plt.loglog(freqs, conform, '.-', label='formula')
plt.legend()
plt.xlabel('freq')
plt.ylabel('scattering matrix cond num')
plt.savefig('figs/freq/highfreq_condnum.png')
plt.clf()


