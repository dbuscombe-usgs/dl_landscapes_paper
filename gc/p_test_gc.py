## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

#general
from __future__ import division
from joblib import Parallel, delayed, cpu_count
import os, time
from glob import glob
from scipy.misc import imread
import itertools

#numerical
import tensorflow as tf
import numpy as np
from scipy.io import savemat
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels, unary_from_softmax
from sklearn.metrics import precision_recall_fscore_support

#plots
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

#supress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

# suppress divide and invalid warnings
np.seterr(divide='ignore')
np.seterr(invalid='ignore')


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
       plt.xticks(tick_marks, classes, fontsize=3) # rotation=45
       plt.yticks(tick_marks, classes, fontsize=3)

       #plt.ylabel('True label',fontsize=6)
       #plt.xlabel('Estimated label',fontsize=6)

    else:
       plt.axis('off')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j]>0:
           plt.text(j, i, format(cm[i, j], fmt),
                 fontsize=3,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    return cm


# =========================================================
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

# =========================================================
def getCP(tmp, graph):  
   #graph = load_graph(classifier_file)

   input_name = "import/Placeholder" ##+ input_layer
   output_name = "import/final_result" ##+ output_layer
   input_operation = graph.get_operation_by_name(input_name);
   output_operation = graph.get_operation_by_name(output_name);

   with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: np.expand_dims(tmp, axis=0)})
   results = np.squeeze(results)

   # Sort to show labels of first prediction in order of confidence
   top_k = results.argsort()[-len(results):][::-1]

   return top_k[0], results[top_k[0]], results[top_k] #, np.std(tmp[:,:,0])


# =========================================================
def norm_im(image_path):
   input_mean = 0 #128
   input_std = 255 #128

   input_name = "file_reader"
   output_name = "normalized"
   img = imread(image_path)
   nx, ny, nz = np.shape(img)

   theta = np.std(img).astype('int')

   file_reader = tf.read_file(image_path, input_name)
   image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
   float_caster = tf.cast(image_reader, tf.float32)

   dims_expander = tf.expand_dims(float_caster, 0);
   normalized = tf.divide(tf.subtract(dims_expander, [input_mean]), [input_std])
   sess = tf.Session()
   return np.squeeze(sess.run(normalized))
      

def eval_tiles(label, direc, numero, classifier_file, x):
#for label in labels:
   graph = load_graph(classifier_file)

   print(label)
   infiles = glob(direc+os.sep+label+os.sep+'*.jpg')[:numero]

   Z = []
   for image_path in infiles:
      Z.append(norm_im(image_path))

   #Z = Parallel(n_jobs=-1, verbose=10)(delayed(norm_im)(image_path) for image_path in infiles) 

   w1 = []
   for i in range(len(Z)):
      w1.append(getCP(Z[i], graph))

   C, P, _ = zip(*w1) ##,S
   del w1, Z

   C = np.asarray(C)
   P = np.asarray(P)

   e = precision_recall_fscore_support(np.ones(len(C))*x, C)

   cm = np.zeros((4,4))
   for a, p in zip(np.ones(len(C), dtype='int')*x, C):
       cm[a][p] += 1

   cm = cm[x,:]

   p = np.max(e[0])
   r = np.max(e[1])
   f = np.max(e[2])
   a = np.sum([c==x for c in C])/len(C)
   #print(label+' accuracy %f' % (a))
   #print('precision %f' % (p) )
   #print('recall %f' % (r) )
   #print('f score %f' % (f) )
   #print('mean prob. %f' % (np.mean(P)) )
   return [a,p,r,f, np.mean(P)], cm ##C,P

#==============================================================
if __name__ == '__main__':

   tile = n = 224 #96
   numero = 1000 

   direc='test/tile_224'

   #=============================================
   class_file = 'labels.txt'

   ## Loads label file, strips off carriage return
   labels = [line.rstrip() for line 
                in tf.gfile.GFile(class_file)]

   code= {}
   for label in labels:
      code[label] = [i for i, x in enumerate([x.startswith(label) for x in labels]) if x].pop()

   #classifier_file = 'gc_mobilenetv2_96_1000_001.pb'
   classifier_file = 'gc_mobilenetv2_224_1000_001.pb'

   w = Parallel(n_jobs=-1, verbose=10)(delayed(eval_tiles)(label, direc, numero, classifier_file, code[label]) for label in labels)
   
   E, CM = zip(*w)
   
   # E = []; CM = []
   # for label in labels:
      # _, _, e, cm = eval_tiles(label, direc, numero, classifier_file, code[label])
      # E.append(e)
      # CM.append(cm)

   CM = np.asarray(CM)

   fig = plt.figure()
   ax1 = fig.add_subplot(221)
   plot_confusion_matrix2(CM, classes=labels, normalize=True, cmap=plt.cm.Reds)
   plt.savefig('gc_cm_224.png', dpi=300, bbox_inches='tight')
   del fig; plt.close()

   a=np.asarray(E)[:,0] 
   f= np.asarray(E)[:,3] 
   pr= np.asarray(E)[:,4] 

   print('mean accuracy. %f' % (np.mean(a)) )
   print('mean Fscore. %f' % (np.mean(f)) )
   print('mean prob. %f' % (np.mean(pr)) )

