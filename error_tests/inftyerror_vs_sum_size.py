import numpy as np
import matplotlib.pyplot as plt
from Ncyl.Ncyl import cyl, scattering

dists = [1.5, 1, 0.8, 0.5, 0.4]
for d in dists:
    radius = 0.3
    M_sums = np.linspace(5, 30, num = 25).tolist()
    M_sums = [int(m) for m in M_sums]
    O1, O2 = np.array([0, d]), np.array([0, -d])
    errs = np.zeros(len(M_sums))

    pts = np.linspace(-3, 3, num = 200)
    X, Y = np.meshgrid(pts, pts)

    u_ref = scattering([cyl(0, 'n', pos = O1, radius = radius),cyl(1, 'd', pos = O2, radius = radius)], M_sum = 100).make_u(X, Y)
    print('u_ref done')
    u_tmp = np.zeros(u_ref.shape)
    for M in M_sums:
        idx = M_sums.index(M)
        u_tmp = scattering([cyl(0, 'n', pos = O1, radius = radius),cyl(1, 'd', pos = O2, radius = radius)], M_sum = M).make_u(X, Y)
        errs[idx] = np.amax(np.abs(u_tmp - u_ref))

    title = 'radius = ' + str(radius) + ', spacing = ' + str(2*d)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.loglog(M_sums, errs, '.-')
    ax1.hlines(np.finfo(float).eps, M_sums[0], M_sums[-1], label = r'$\epsilon_{mach}$')
    ax1.set_ylabel(r'$||u_{ref} - u_M||_\infty$')
    ax1.set_xlabel('sum index')
    ax1.set_title('loglog')
    ax1.legend()
    ax2.semilogy(M_sums, errs, '.-')
    ax2.hlines(np.finfo(float).eps, M_sums[0], M_sums[-1], label = r'$\epsilon_{mach}$')
    ax2.set_xlabel('sum index')
    ax2.set_title('semilog')
    ax2.legend()
    fig.suptitle(title)
    fig.set_size_inches(9, 4)
    fig.set_dpi(200)
    #plt.show()  
    plt.savefig('results/'+title+'.png')
