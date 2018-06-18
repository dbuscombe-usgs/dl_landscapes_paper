## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

from __future__ import division

import sys, getopt, os
import cv2
import numpy as np
from scipy.misc import imsave #imresize, imread
from scipy.io import savemat

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from numpy.lib.stride_tricks import as_strided as ast
#from skimage.morphology import remove_small_objects
from scipy.stats import mode as md
import random, string
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels, unary_from_softmax
from joblib import Parallel, delayed, cpu_count

import dask.bag as db
from tempfile import mkdtemp
import os.path as path


# =========================================================
def writeout(tmp, cl, labels, outpath, thres):

   l, cnt = md(cl.flatten())
   l = np.squeeze(l)
   if cnt/len(cl.flatten()) > thres:
      outfile = id_generator()+'.jpg'
      outpath = outpath+labels[l]+'/'+outfile
      imsave(outpath, tmp)

# =========================================================
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))

# =========================================================
def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')


# =========================================================
# Return a sliding window over a in any number of dimensions
# version with no memory mapping
def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    '''
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)

    return a.reshape(dim), newshape


# =========================================================
# Return a sliding window over a in any number of dimensions
def sliding_window_mm(a,ws,ss = None,flatten = True):
   '''
   Return a sliding window over a in any number of dimensions
   '''
   if None is ss:
      # ss was not provided. the windows will not overlap in any direction.
      ss = ws
   ws = norm_shape(ws)
   ss = norm_shape(ss)
   # convert ws, ss, and a.shape to numpy arrays
   ws = np.array(ws)
   ss = np.array(ss)

   shape_a = np.shape(a)
   dtype_a = a.dtype
   filename = path.join(mkdtemp(), 'tmp.dat')
   fp = np.memmap(filename, dtype=dtype_a, mode='w+', shape=shape_a)

   fp[:] = a[:]
   del a

   a = np.memmap(filename, dtype=dtype_a, mode='r', shape=shape_a)

   shap = np.array(shape_a)

   # ensure that ws, ss, and a.shape all have the same number of dimensions
   ls = [len(shap),len(ws),len(ss)]
   if 1 != len(set(ls)):
      raise ValueError(\
      'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

   # ensure that ws is smaller than a in every dimension
   if np.any(ws > shap):
      raise ValueError(\
      'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

   # how many slices will there be in each dimension?
   newshape = norm_shape(((shap - ws) // ss) + 1)
   # the shape of the strided array will be the number of slices in each dimension
   # plus the shape of the window (tuple addition)
   newshape += norm_shape(ws)
   # the strides tuple will be the array's strides multiplied by step size, plus

   try:

      # the array's strides (tuple addition)
      newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
      a = ast(a,shape = newshape,strides = newstrides)
      if not flatten:
         return a
      # Collapse strided so that it has one more dimension than the window.  I.e.,
      # the new array is a flat list of slices.
      meat = len(ws) if ws.shape else 0
      firstdim = (int(np.product(newshape[:-meat])),) if ws.shape else ()
      dim = firstdim + (newshape[-meat:])
      # remove any dimensions with size 1
      #dim = filter(lambda i : i != 1,dim)

      return a.reshape(dim), newshape

   except:

      from itertools import product
      # For each dimension, create a list of all valid slices
      slices = [[] for i in range(len(ws))]
      for i in range(len(ws)):
         nslices = ((shap[i] - ws[i]) // ss[i]) + 1
         for j in range(0,nslices):
            start = j * ss[i]
            stop = start + ws[i]
            slices[i].append(slice(start,stop))
      # Get an iterator over all valid n-dimensional slices of the input
      allslices = product(*slices)

      # Allocate memory to hold all valid n-dimensional slices
      nslices = np.product([len(s) for s in slices])
      #out = np.ndarray((nslices,) + tuple(ws),dtype = a.dtype)
      out=[]
      for i,s in enumerate(allslices):
         #out[i] = a[s]
         out.append(a[s])

      del a
      tmp = db.from_sequence(out, npartitions=1000)
      del out

      return tmp.compute(), newshape



# =========================================================
def getCRF_justcol(img, Lc, theta, n_iter, label_lines, compat_col=40, scale=5, prob=0.5):

      H = img.shape[0]
      W = img.shape[1]

      d = dcrf.DenseCRF2D(H, W, len(label_lines)+1)
      U = unary_from_labels(Lc.astype('int'), len(label_lines)+1, gt_prob= prob)

      d.setUnaryEnergy(U)

      del U

      # sdims = The scaling factors per dimension.
      # schan = The scaling factors per channel in the image.
      # This creates the color-dependent features and then add them to the CRF
      feats = create_pairwise_bilateral(sdims=(theta, theta), schan=(scale, scale, scale), #11,11,11
                                  img=img, chdim=2)

      del img

      d.addPairwiseEnergy(feats, compat=compat_col,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

      del feats
      Q = d.inference(n_iter)

      preds = np.array(Q, dtype=np.float32).reshape(
        (len(label_lines)+1, H, W)).transpose(1, 2, 0)
      preds = np.expand_dims(preds, 0)
      preds = np.squeeze(preds)

      return np.argmax(Q, axis=0).reshape((H, W)), preds #, p, R, np.abs(d.klDivergence(Q)/ (H*W))



#==============================================================================

# mouse callback function
def anno_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode, lw

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),lw) #10, 5)
                current_former_x = former_x
                current_former_y = former_y
                #print former_x,former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),lw) #5)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y


#==============================================================================
#==============================================================================

#==============================================================
if __name__ == '__main__':

   argv = sys.argv[1:]
   try:
      opts, args = getopt.getopt(argv,"hi:w:")
   except getopt.GetoptError:
      print('python int_seg_crf.py -i image')
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print('Example usage: python int_seg_crf.py -i J:\Elwha\Elwha_20120927\Elwha_20120927_2500_25000.tif -w 1000')
         sys.exit()
      elif opt in ("-i"):
         image_path = arg
      elif opt in ("-w"):
         win = arg

   #===============================================

   win = int(win) ##1000

   lw = 10 #initial brush thickness
   print("initial brush width = "+str(lw))
   print("change using the +/- keys")
   
   theta=300 #100
   compat_col=100 #40
   scale=1
   n_iter=10 #20

   thres = .9
   tile = 128   


   labels = ['terrain', 'cliff','water','veg','sky','foam','sand', 'anthro','road']
   cmap = ['#D2691E', '#8B008B','b','g','c','w','#FFD700','r', '#696969']

   classes = dict(zip(labels, cmap))

   cmap = colors.ListedColormap(cmap)



   outpath = 'autoclassified_seabright'+str(tile)
   #=======================================================
   try:
      os.mkdir(outpath)
   except:
      pass

   for f in classes.keys():
      try:
         os.mkdir(outpath+os.sep+f)
      except:
         pass


   #===============================================

   drawing=False # true if mouse is pressed
   mode=True # if True, draw rectangle. Press 'm' to toggle to curve

   img = cv2.imread(image_path)

   img[img==0] = 1
   img[img==255] = 254

   # if not img: #if image is empty, we'll assume it is a geotiff   

      # import rasterio
      # print('Reading GeoTIFF data ...')
      # input = [image_path]

      # ## read all arrays
      # bs = []
      # for layer in input:
         # with rasterio.open(layer) as src:
            # layer = src.read()[0,:,:]
         # w, h = (src.width, src.height)
         # xmin, ymin, xmax, ymax = src.bounds
         # crs = src.get_crs()
         # del src
         # bs.append({'bs':layer, 'w':w, 'h':h, 'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax, 'crs':crs})

      # img = np.dstack([x['bs'] for x in bs]).astype('uint8')

   #mask=img[:,:,0]<20
   #img[mask] = 0

   nxo, nyo, nz = np.shape(img)
   # pad image so it is divisible by N windows with no remainder
   img = np.pad(img, [(0,win-np.mod(nxo,win)), (0,win-np.mod(nyo,win)), (0,0)], mode='constant')
   nx, ny, nz = np.shape(img)

   try:
      print("trying memory mapping")
      Z,ind = sliding_window_mm(img, (win, win,3), (win, win,3))
   except:
      print("memory mapping failed")
      Z,ind = sliding_window(img, (win, win,3), (win, win,3))


   gridy, gridx = np.meshgrid(np.arange(ny), np.arange(nx))

   try:
      print("trying memory mapping")
      Zx,_ = sliding_window_mm(gridx, (win, win), (win, win))
      Zy,_ = sliding_window_mm(gridy, (win, win), (win, win))
   except:
      print("memory mapping failed")
      Zx,_ = sliding_window(gridx, (win, win), (win, win))
      Zy,_ = sliding_window(gridy, (win, win), (win, win))

   out = np.zeros((nx,ny))

   for ck in range(len(Z)):

      im = Z[ck]
      counter=1
      if np.std(im)>0:
         for label in labels:

            cv2.namedWindow(label)#, cv2.WND_PROP_FULLSCREEN) 
            cv2.moveWindow(label, 0,0)  # Move it to (0,0)
            #cv2.setWindowProperty(label, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(label,anno_draw)
            while(1):
               cv2.imshow(label,im)
               k=cv2.waitKey(1)&0xFF
               if k==27:
                  im[im[:,:,2]==255] = counter
                  counter += 1
                  break
               #plus = 43
               if k==43:
                  lw += 1
                  print("brush width = "+str(lw))
               #minus = 45
               if k==45:
                  lw -= 1
                  if lw<1:
                     lw=1
                  print("brush width = "+str(lw))
				  
            cv2.destroyAllWindows()

         Lc = im[:,:,2]
         Lc[Lc>=counter] = 0

         out[Zx[ck],Zy[ck]] = Lc
      else:
         Lc = np.zeros(np.shape(im[:,:,2]))
         out[Zx[ck],Zy[ck]] = Lc


   Lc = out[:nxo,:nyo] ##np.round(imresize(Lc,np.shape(im), interp='nearest'))

   im = img[:nxo, :nyo]

   Lcorig = Lc.copy().astype('float')
   Lcorig[Lcorig<1] = np.nan

   print('Generating dense scene from sparse labels ....')
   res,p = getCRF_justcol(im, Lc, theta, n_iter, classes, compat_col, scale)

   savemat(image_path.split('.')[0]+'_mres.mat', {'sparse': Lc.astype('int'), 'class': res.astype('int'), 'preds': p.astype('float16'), 'labels': labels}, do_compression = True)

   #=============================================
   name, ext = os.path.splitext(image_path)
   name = name.split(os.sep)[-1]
   print('Generating plot ....')
   fig = plt.figure()
   fig.subplots_adjust(wspace=0.4)
   ax1 = fig.add_subplot(131)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   _ = ax1.imshow(im)
   plt.title('a) Input', loc='left', fontsize=6)

   ax1 = fig.add_subplot(132)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   _ = ax1.imshow(im)
   plt.title('b) Unary potentials', fontsize=6)
   im2 = ax1.imshow(Lcorig-1, cmap=cmap, alpha=0.5, vmin=0, vmax=len(labels))
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
   plt.title('c) CRF prediction', loc='left', fontsize=6)
   im2 = ax1.imshow(res, cmap=cmap, alpha=0.5, vmin=0, vmax=len(labels))
   divider = make_axes_locatable(ax1)
   cax = divider.append_axes("right", size="5%")
   cb=plt.colorbar(im2, cax=cax)
   cb.set_ticks(np.arange(len(labels)+1))
   cb.ax.set_yticklabels(labels)
   cb.ax.tick_params(labelsize=4)
   plt.savefig(name+'_mres.png', dpi=600, bbox_inches='tight')
   del fig; plt.close()

   #==============================

   print('Generating tiles from dense class map ....')
   #Z,ind = sliding_window(im, (tile,tile,3), (int(tile/2), int(tile/2),3))
   #C,ind = sliding_window(res, (tile,tile), (int(tile/2), int(tile/2)))

   try:
      print("trying memory mapping")
      Z,ind = sliding_window_mm(im, (tile,tile,3), (int(tile/2), int(tile/2),3))
      C,ind = sliding_window_mm(res, (tile,tile), (int(tile/2), int(tile/2)))	  
   except:
      print("memory mapping failed")
      Z,ind = sliding_window(im, (tile,tile,3), (int(tile/2), int(tile/2),3))
      C,ind = sliding_window(res, (tile,tile), (int(tile/2), int(tile/2)))
   

   w = Parallel(n_jobs=-1, verbose=0, pre_dispatch='2 * n_jobs', max_nbytes=None)(delayed(writeout)(Z[k], C[k], labels, outpath+os.sep, thres) for k in range(len(Z)))
