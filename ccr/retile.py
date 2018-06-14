## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

#general
from __future__ import division
from joblib import Parallel, delayed
from glob import glob
import numpy as np 
from scipy.misc import imread
from scipy.io import loadmat
import sys, getopt, os

from tile_utils import *

from scipy.stats import mode as md
from scipy.misc import imsave

# =========================================================
def writeout(tmp, cl, labels, outpath, thres):

   l, cnt = md(cl.flatten())
   l = np.squeeze(l)
   if cnt/len(cl.flatten()) > thres:
      outfile = id_generator()+'.jpg'
      fp = outpath+os.sep+labels[l]+os.sep+outfile
      imsave(fp, tmp)

#==============================================================
if __name__ == '__main__':

   direc = ''; tile = ''; thres = ''

   argv = sys.argv[1:]
   try:
      opts, args = getopt.getopt(argv,"hi:t:a:")
   except getopt.GetoptError:
      print('python retile.py -i direc -t tilesize -a threshold')
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print('Example usage: python retile.py -i train -t 96 -a 0.9')
         sys.exit()
      elif opt in ("-i"):
         direc = arg
      elif opt in ("-t"):
         tile = arg
      elif opt in ("-a"):
         thres = arg

   if not direc:
      direc = 'train'
   if not tile:
      tile = 96
   if not thres:
      thres = .9

   tile = int(tile)
   thres = float(thres)

   #=======================================================
   outpath = direc+os.sep+'tile_'+str(tile)
   files = sorted(glob(direc+os.sep+'*.mat'))

   #labels = loadmat(files[0])['labels']
   labels = ['surf', 'buildings','sky','terrain','water','veg','swash','beach','road','cliff']

   labels = [label.replace(' ','') for label in labels]
   #=======================================================

   #=======================================================
   try:
      os.mkdir(outpath)
   except:
      pass

   for f in labels:
      try:
         os.mkdir(outpath+os.sep+f)
      except:
         pass
   #=======================================================

   #=======================================================
   for f in files:

      dat = loadmat(f)
      res = dat['class']
	  
      #labels = dat['labels']
      #labels = [label.replace(' ','') for label in labels]	  
	  
      fim = direc+os.sep+f.split(os.sep)[-1].replace('_class.mat','')

      print('Generating tiles from dense class map ....')
      Z,ind = sliding_window(imread(fim), (tile,tile,3), (int(tile/2), int(tile/2),3)) 

      C,ind = sliding_window(res, (tile,tile), (int(tile/2), int(tile/2))) 

      w = Parallel(n_jobs=-1, verbose=0, pre_dispatch='2 * n_jobs', max_nbytes=None)(delayed(writeout)(Z[k], C[k], labels, outpath, thres) for k in range(len(Z))) 
	  
      # for k in range(len(Z)):
         # writeout(Z[k], C[k], labels, outpath, thres)	  


