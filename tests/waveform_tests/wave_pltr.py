import numpy as np
import matplotlib.pyplot as plt
from waveforms import waveforms as wvf

from matplotlib.widgets import Button, Slider

N = 100
x = np.linspace(-10, 10, N)
y = np.linspace(-10, 10, N)
X, Y = np.meshgrid(x, y)

t = 0
T = 20
wave = wvf.gaussian(time_shift=5,
                    inc_angle=-np.pi/3, 
                    src_type='pointsrc')

fig, ax = plt.subplots()
def replot(t):
    wvplt = wave.td_wave(X, Y, t)
    CS = ax.pcolormesh(X, Y, wvplt)
    #ax.clabel(CS)
    fig.subplots_adjust(left=0.25, bottom=0.25)

replot(t)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
ang_slider = Slider(
    ax=axfreq,
    label='t',
    valmin=0,
    valmax=T,
    valinit=0,
)

# The function to be called anytime a slider's value changes
def update(val):
    t = ang_slider.val
    replot(t)
    fig.canvas.draw_idle()

# register the update function with each slider
ang_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    ang_slider.reset()
button.on_clicked(reset)

plt.show()

