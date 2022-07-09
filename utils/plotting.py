import numpy as np
import matplotlib.pyplot as plt

def plot_soln(x, y, sc, title, fname, dpi=200):
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
    #plt.show()
    plt.savefig(fname, dpi = dpi)


