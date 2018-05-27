
import numpy as np
from scipy.misc import imsave, imread
from scipy.io import loadmat

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors


labels = ['terrain', 'cliff','water','veg','sky','foam','sand', 'anthro','road']
cmap = ['#D2691E', '#8B008B','b','g','c','w','#FFD700','r', '#696969']

classes = dict(zip(labels, cmap))

cmap = colors.ListedColormap(cmap)

imfiles = [r'test\usgs_pcmsc_2016_02_05_223923.TIF-0.jpg']

resfiles = [r'test\usgs_pcmsc_2016_02_05_223923_mres.mat']

for k in range(len(imfiles)):

   dat = loadmat(resfiles[k])
   im = imread(imfiles[k])   
		   
   fig = plt.figure()
   fig.subplots_adjust(wspace=0.4)
   ax1 = fig.add_subplot(131)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   _ = ax1.imshow(im)
   plt.title('a) Input', loc='left', fontsize=5)

   ax1 = fig.add_subplot(132)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   _ = ax1.imshow(im)
   plt.title('b) Unary potentials', fontsize=5, loc='left')
   im2 = ax1.imshow(dat['sparse']-1, cmap=cmap, alpha=0.5, vmin=0, vmax=len(labels))
   divider = make_axes_locatable(ax1)
   cax = divider.append_axes("right", size="5%")
   cb=plt.colorbar(im2, cax=cax)
   cb.set_ticks(np.arange(len(labels)+1))
   cb.ax.set_yticklabels(labels)
   cb.ax.tick_params(labelsize=4)

   ax1 = fig.add_subplot(133)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   _ = ax1.imshow(im)
   plt.title('c) CRF prediction', loc='left', fontsize=5)
   im2 = ax1.imshow(dat['class'], cmap=cmap, alpha=0.5, vmin=0, vmax=len(labels))
   divider = make_axes_locatable(ax1)
   cax = divider.append_axes("right", size="5%")
   cb=plt.colorbar(im2, cax=cax)
   cb.set_ticks(np.arange(len(labels)+1))
   cb.ax.set_yticklabels(labels)
   cb.ax.tick_params(labelsize=4)
   plt.savefig('seabright_mres_ex.png', dpi=600, bbox_inches='tight')
   del fig; plt.close()
