import sys
from Ncyl.Ncyl import scattering, cyl
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

def plot_soln(x, y, sc, title, fname, outp_soln_to_text=False, text_soln_fname = ''):
    xl = np.amin(x)
    xr = np.amax(x)
    yl = np.amin(y)
    yr = np.amax(y)

    fig, ax = plt.subplots()

    X, Y = np.meshgrid(x,y)
    Z = sc.make_u(X,Y)
    if outp_soln_to_text:
        np.save(text_soln_fname, Z)
    im = ax.imshow(np.abs(Z), extent=[xl, xr, yl, yr], origin = 'lower', cmap='viridis')
    ax.set_title(title)

    #mark the circles according to BC
    angs = np.linspace(0, 2*np.pi, num = 20)

    for i in sc.labels:
        bc = sc.cyls[i].bc
        color = 'b'
        if bc == 'd':
            color = 'g'
        elif bc == 'i':
            color = 'w'
        elif bc =='n':
            color = 'm'
        a = sc.cyls[i].radius
        ax.plot(a*np.cos(angs) + sc.cyls[i].pos[0], a*np.sin(angs) + sc.cyls[i].pos[1], color=color, label='bc = '+sc.cyls[i].bc)

    #needed to get rid of duplicates in the plot legend labels
    #ripped from https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

    legend_without_duplicate_labels(ax)
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    #plt.show()
    plt.savefig(fname, dpi = 300)

b = 3
radius = 0.3

sc = scattering([cyl(0, 'n', pos = np.array([0, b/2]), radius = radius),cyl(1, 'd', pos = np.array([0, -b/2]), radius = radius)], M_sum = 100)
grid_num = 200
pts = np.linspace(-3, 3, num = grid_num)
plot_soln(pts, pts, sc, 'radius = 0.3, dist from origin = 1.5', 'ref_sol_M=100.png', outp_soln_to_text=True, text_soln_fname= 'ref_sol_M=100.npy')


