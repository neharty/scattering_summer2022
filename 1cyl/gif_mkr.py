import imageio
import os

images = []
fnames = os.listdir('1cyl_gif_imgs')
fnames.sort()
for fname in fnames:
    images.append(imageio.imread(os.path.join('1cyl_gif_imgs',fname)))
imageio.mimsave('1cyl.gif', images, duration = '0.02')

