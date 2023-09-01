import pickle
from waveforms.waveforms import gaussian
from utils import example_setups

outpfile = 'wave.p'
wv = gaussian(time_shift=6)
pickle.dump(wv, open(outpfile, 'wb'))