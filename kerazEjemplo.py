

import numpy as np  
from scipy import misc  
from PIL import Image  
import glob  
import matplotlib.pyplot as plt  
import scipy.misc  
from matplotlib.pyplot import imshow  
# %matplotlib inline
import tensorflow as tf 
from IPython.display import SVG  
import cv2  
import seaborn as sn  
import pandas as pd  
import pickle  
from keras import layers
from keras.layers import Flatten, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout  
from keras.models import Sequential, Model, load_model  
from keras.preprocessing import image  
from keras.preprocessing.image import load_img  
from keras.preprocessing.image import img_to_array  
from keras.applications.imagenet_utils import decode_predictions  
from keras.utils import layer_utils, np_utils  
from keras.utils.data_utils import get_file  
from keras.applications.imagenet_utils import preprocess_input  
from keras.utils.vis_utils import model_to_dot  
# from keras.utils import plot_model  
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform  
from keras import losses  
import keras.backend as K  
from keras.callbacks import ModelCheckpoint  
from sklearn.metrics import confusion_matrix, classification_report  
 
from keras.datasets import cifar100



(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar100.load_data(label_mode='fine')

y_train = np_utils.to_categorical(y_train_original, 100)
y_test = np_utils.to_categorical(y_test_original, 100)
imgplot = plt.imshow(x_train_original[3])
plt.show()

# lo que haremos a continuación, es normalizar las imágenes. Esto es, 
# dividiremos cada elemento de x_train_original por el numero de píxeles, 
# es decir, 255. Con esto obtenemos que el array comprenderá valores de entre 0 y 1. 
# Con esto el entrenamiento suele aportar mejores resultados.
x_train = x_train_original/255
x_test = x_test_original/255

# El siguiente paso es definir ciertos parametros sobre el experimento en Keras. 
# Lo primero será especificar a Keras dónde se encuentran los canales. En un array de imagenes, 
# pueden venir como ultimo indice o como el primero. Esto se conoce como canales primero (channels first) 
# o canales al final (channels last). En nuestro caso, vamos a definirlos al final.

K.set_image_data_format('channels_last')

# Lo siguiente que vamos a especificar es la fase del experimento. 
# En este caso, la fase será de entrenamiento.
K.set_learning_phase(1)


# En primer lugar, vamos a entrenar una red neuronal sencilla. 
# Definimos un procedimiento que nos devuelva una red neuronal.
def create_simple_nn():
  model = Sequential()
  model.add(Flatten(input_shape=(32, 32, 3), name="Input_layer"))
  model.add(Dense(1000, activation='relu', name="Hidden_layer_1"))
  model.add(Dense(500, activation='relu', name="Hidden_layer_2"))
  model.add(Dense(100, activation='softmax', name="Output_layer"))
  
  return model

snn_model = create_simple_nn()
snn_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', 'mse'])

snn_model.summary()


# Podemos ver que, para ser un modelo simple de red neuronal,
# tiene que entrenar más de 3 millones de parámetros. 
# Esta será la razón por la que existe el aprendizaje profundo,
# ya que para entrenar redes muy complejas se necesitaría entrenar
# de esta forma grandes cantidades de parámetros.

# Ahora sólo queda entrenar, para ello, haremos lo siguiente:
snn = snn_model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test), shuffle=True)

# snn = snn_model.evaluate(x=x_test, y=y_test, batch_size=32, verbose=1)
# evaluation
# Veamos las métricas obtenidas para el entrenamiento
# y validación gráficamente (para ello usamos la librería matplotlib)

# plt.figure(0)
# plt.plot(snn.history['acc'],'r')
# plt.plot(snn.history['val_acc'],'g')
# plt.xticks(np.arange(0, 11, 2.0))
# plt.rcParams['figure.figsize'] = (8, 6)
# plt.xlabel("Num of Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training Accuracy vs Validation Accuracy")
# plt.legend(['train','validation'])
 
# plt.figure(1)
# plt.plot(snn.history['loss'],'r')
# plt.plot(snn.history['val_loss'],'g')
# plt.xticks(np.arange(0, 11, 2.0))
# plt.rcParams['figure.figsize'] = (8, 6)
# plt.xlabel("Num of Epochs")
# plt.ylabel("Loss")
# plt.title("Training Loss vs Validation Loss")
# plt.legend(['train','validation'])
 
# plt.show()


# Una vez que hemos entrenado el modelo, vamos a ver otras métricas. 
# Para ello, crearemos la matriz de confusión y, a partir de ella, 
# veremos las métricas precission, recall y F1-score (ver wikipedia).

# Vamos a hacer una predicción sobre el dataset de validación y,
# a partir de ésta, generamos la matriz de confusión 
# y mostramos las métricas mencionadas anteriormente.



snn_pred = snn_model.predict(x_test, batch_size=32, verbose=1)
snn_predicted = np.argmax(snn_pred, axis=1)

#Creamos la matriz de confusión
snn_cm = confusion_matrix(np.argmax(y_test, axis=1), snn_predicted)

# Visualiamos la matriz de confusión
snn_df_cm = pd.DataFrame(snn_cm, range(100), range(100))
plt.figure(figsize = (20,14))
sn.set(font_scale=1.4) #for label size
sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 6}) # font size
plt.show()


# Y por último, mostramos las métricas
snn_report = classification_report(np.argmax(y_test, axis=1), snn_predicted)
print(snn_report)













from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

n_classes = 100

from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], snn_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), snn_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# # Plot all ROC curves
# plt.figure(1)
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)

# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes-97), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()


# # Zoom in view of the upper left corner.
# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)

# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(3), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()



imgplot = plt.imshow(x_train_original[0])
plt.show()
print('class for image 1: ' + str(np.argmax(y_test[0])))
print('predicted:         ' + str(snn_predicted[0]))


imgplot = plt.imshow(x_train_original[3])
plt.show()
print('class for image 1: ' + str(np.argmax(y_test[3])))
print('predicted:         ' + str(snn_predicted[3]))

#Histórico
with open('simplenn_history.txt', 'wb') as file_pi:
  pickle.dump(snn.history, file_pi)