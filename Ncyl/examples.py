from Ncyl import cyl, scattering
import numpy as np
import matplotlib.pyplot as plt

#plotting stuff
def plot_soln(x, y, sc, title, fname):
    xl = np.amin(x)
    xr = np.amax(x)
    yl = np.amin(y)
    yr = np.amax(y)
    
    fig, ax = plt.subplots()

    X, Y = np.meshgrid(x,y)
    im = ax.imshow(np.abs(sc.make_u(X,Y)), extent=[xl, xr, yl, yr], origin = 'lower', cmap='viridis')
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
    plt.show()
    #plt.savefig(fname, dpi = 300)

def grid_of_cyls():
    # 3x3 grid of cylinders
    cyls = [None for i in range(9)]
    radius = 0.2
    space = 1.5
    origins = np.array([[-space, -space], [-space, 0], [-space, space], 
            [0, -space], [0, 0],[0, space], 
            [space, -space], [space, 0], [space, space]])
    for i in range(9):
        origins[i] = np.array(origins[i])
    print(origins)

    for i in range(9):
        bc = 'd'
        if i%3 == 0:
            bc = 'd' 
        elif i%3 == 1:
            bc = 'n'
        elif i%3 == 2:
            bc = 'i'
        else:
            None

        cyls[i] = cyl(i, bc, pos=origins[i], radius=radius)

    return scattering(cyls)


def pentagram():
    origins_cmplx = 1j*np.zeros(5)
    for i in range(5):
        origins_cmplx[i] = 2*np.exp(1j*2*np.pi*i/5)
        
    origins = np.zeros((5, 2))
    
    origins[:, 0] = np.real(origins_cmplx)[:]
    origins[:, 1] = np.imag(origins_cmplx)[:]
    
    print(origins)
    bcs = ['d', 'n', 'i', 'd', 'n']
    radius = 0.3
    return scattering([cyl(i, bcs[i], pos=origins[i], radius = radius) for i in range(5)])

def main():
    xl = -3
    xr = 3
    yl = -3
    yr = 3
    N_points = 200
    x = np.linspace(xl, xr, num = N_points)
    y = np.linspace(yl, yr, num = N_points)
    
    #plot_soln(x, y, grid_of_cyls(), 'radius = 0.2, spacing = 1.5', 'examples/9cyls.png')
    #plot_soln(x, y, pentagram(), 'radius = 0.3, spacing = 2', 'examples/pentagram.png')
    
    plot_soln(x, y, scattering([cyl(0, 'd', pos = np.array([-1, -1]), radius = 0.3)]), '', '')

main()
