import numpy as np
import matplotlib.pyplot as plt

def N_p(ka, tol): 
    return np.array([int(ka[i] + (np.log(2*np.sqrt(2)*np.pi*ka[i]/tol))**(2/3)*(ka[i])**(1/3) + 1) for i in range(len(ka))])

kas = np.logspace(-1, 2, num = 200)

tols = [1e-7, 1e-9, 1e-11]
lss = ['-', '--', '-.']
cols = ['k', 'r', 'b']

for i in range(3):
    plt.loglog(kas, N_p(kas, tols[i]), color=cols[i], ls = lss[i], label='eps = ' + str(tols[i]))
    
plt.title('N_p vs ka from Antoine, using 200 points')
plt.xlabel('ka')
plt.ylabel('N_p')
plt.legend()
plt.savefig('antoine_N_p_vs_ka.png', dpi = 200)

