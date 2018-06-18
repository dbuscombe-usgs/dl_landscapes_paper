## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

#general
from __future__ import division
import os, time
from glob import glob
from scipy.misc import imread
import itertools

#numerical
import numpy as np
from scipy.io import savemat, loadmat
from sklearn.metrics import precision_recall_fscore_support

from skimage.morphology import disk
from skimage.filters.rank import median
from skimage.measure import label, regionprops

#plots
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors


## =========================================================
def plot_confusion_matrix2(cm, classes, normalize=False, cmap=plt.cm.Blues, dolabels=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)] = 0

    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmax=1, vmin=0)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    if dolabels==True:
       tick_marks = np.arange(len(classes))
       plt.xticks(tick_marks, classes, fontsize=4) # rotation=45
       plt.yticks(tick_marks, classes, fontsize=4)

    else:
       plt.axis('off')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j]>0:
           plt.text(j, i, format(cm[i, j], fmt),
                 fontsize=4,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    return cm



direc_true='test'
direc_predict='ares96'

true_files = sorted(glob(direc_true+os.sep+'*.mat'))
pred_files = sorted(glob(direc_predict+os.sep+'*.mat'))

CM = [] ; P = [] ; R = [] ; F = [] ;  N = [] ;  AR = []

for k in range(len(true_files)):
   print(k)
   pred = loadmat(pred_files[k])['class'].astype('int8')
   pred_labels = loadmat(pred_files[k])['labels']

   true = loadmat(true_files[k])['class'].astype('int8')
   true_labels = loadmat(true_files[k])['labels']

   code = []
   for l in true_labels:
      tmp = [i for i, x in enumerate([x.startswith(l) for x in true_labels]) if x].pop()
      code.append(tmp)
   
   recode = []
   for l in pred_labels:
      tmp = [i for i, x in enumerate([x.startswith(l) for x in true_labels]) if x].pop()
      recode.append(tmp)
	  
   if not len(code)==len(recode):
      recode = recode + np.setdiff1d(code, recode).tolist()   
	  	  
   pred_recode = np.zeros(np.shape(pred), dtype='int8')
   for kk in range(len(code)):
      pred_recode[pred==code[kk]] = recode[kk]   	  
	  
   truef = median(true, disk(5))
   for i,j in zip(np.unique(true), np.unique(truef)):
      truef[truef==j] = i
   
   predf = median(pred_recode, disk(5))
   for i,j in zip(np.unique(pred_recode), np.unique(predf)):
      predf[predf==j] = i

   
   AA = []
   for kk in np.unique(truef):
      label_img = label(truef==kk)
      regions = regionprops(label_img) 
      A = []   
      for props in regions:
         A.append(props.area)
 
      AA.append(A)
	  
   t = truef.flatten()
   p = predf.flatten()
   
   e = precision_recall_fscore_support(t, p)
   pr = e[0]; pr[pr==0] = np.nan   
   r = e[1]; r[r==0] = np.nan   
   f = e[2]; f[f==0] = np.nan  
   
   P.append(pr)   
   R.append(r)   
   F.append(f)   
   N.append(e[3])   
   AR.append(AA)
   
   cm = np.zeros((len(true_labels),len(true_labels)))
   for a, p in zip(t, p):
       cm[a][p] += 1

   CM.append(cm)

AR2 = [x for x in AR if len(x) == len(true_labels)]
   

CM = np.squeeze(np.asarray(CM))
   
fig = plt.figure()
ax1 = fig.add_subplot(221)
plot_confusion_matrix2(np.mean(CM, axis=0), classes=true_labels, normalize=True, cmap=plt.cm.Reds)
plt.savefig('ontario_cm_96_pixclass.png', dpi=300, bbox_inches='tight')
del fig; plt.close()   

P = np.squeeze(np.asarray(P))
R = np.squeeze(np.asarray(R))
F = np.squeeze(np.asarray(F))

print(np.max((np.nanmedian(P, axis=0), np.nanmean(P, axis=0)), axis=0))   
print(np.max((np.nanmedian(R, axis=0), np.nanmean(R, axis=0)), axis=0))   
print(np.max((np.nanmedian(F, axis=0), np.nanmean(F, axis=0)), axis=0))   

print(np.mean(AR2, axis=0))
