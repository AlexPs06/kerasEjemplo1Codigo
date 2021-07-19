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

# Carga del conjunto de datos a usar en este caso es CIFAR-100

(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar100.load_data(label_mode='fine')


#Conversion a one-hot-coding de los conjuntos Y recordad que este conjunto tienes las etiquetas de las clases
y_train = np_utils.to_categorical(y_train_original, 100)  
y_test = np_utils.to_categorical(y_test_original, 100)  


#imgplot = plt.imshow(x_train_original[3])  
#plt.show()  

#Normalización de los datos del conjunto X para su procesamiento
x_train = x_train_original/255  
x_test = x_test_original/255  

#Definicion del canal de las imagenes a usar
K.set_image_data_format('channels_last')  
#indicacion de la fase que estamos realizado 0 para pruebas 1 para entrenamiento
K.set_learning_phase(1)  

#cración de la red convolucional
def create_simple_cnn():  
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu'))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))

  model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(100, activation='softmax'))

  return model

#Compilación del modelo de la red
scnn_model = create_simple_cnn()  
scnn_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', 'mse'])  
#Resumen del modelo creado
scnn_model.summary()  


#Entrenamiento de la red en esta funcion se define los parametros del entrenamiento y los valores con los que se probara
#ademas de las generacion y el tamaño de estas
scnn = scnn_model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test), shuffle=True)  


#evaliucon durante el entrenamiento
# cnn_evaluation = scnn_model.evaluate(x=x_test, y=y_test, batch_size=32, verbose=1)  
# cnn_evaluation  

#Visualizacion de los resultados del entrenamiento
plt.figure(0)  
plt.plot(scnn.history['acc'],'r')  
plt.plot(scnn.history['val_acc'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])

plt.figure(1)  
plt.plot(scnn.history['loss'],'r')  
plt.plot(scnn.history['val_loss'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])

plt.show()  

#Modelo de prediccion basado en el conjunto de entrenamiento
scnn_pred = scnn_model.predict(x_test, batch_size=32, verbose=1)  
scnn_predicted = np.argmax(scnn_pred, axis=1) 


#Creamos la matriz de confusión
scnn_cm = confusion_matrix(np.argmax(y_test, axis=1), scnn_predicted)

# Visualiamos la matriz de confusión
scnn_df_cm = pd.DataFrame(scnn_cm, range(100), range(100))  
plt.figure(figsize = (20,14))  
sn.set(font_scale=1.4) #for label size  
sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12}) # font size  
plt.show()  

#Metricas de precision
scnn_report = classification_report(np.argmax(y_test, axis=1), scnn_predicted)  
print(scnn_report)  


#curva de ROC
from sklearn.datasets import make_classification  
from sklearn.preprocessing import label_binarize  
from scipy import interp  
from itertools import cycle

n_classes = 100

from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2

# Calculo de la curva de ROC y el area de ROC para cada clase
fpr = dict()  
tpr = dict()  
roc_auc = dict()  
for i in range(n_classes):  
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], scnn_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calculo del  micro-media de la curva de ROC  y el ROC area 
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), scnn_pred.ravel())  
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Primero añadimos todos los falsos positivos 
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Despues se interpolan todos las curvas de ROC con los puntos
mean_tpr = np.zeros_like(all_fpr)  
for i in range(n_classes):  
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finalmente promediamos y calculamos el AUC 
mean_tpr /= n_classes

fpr["macro"] = all_fpr  
tpr["macro"] = mean_tpr  
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Pintamos los puntos de todas las curvas de ROC 
plt.figure(1)  
plt.plot(fpr["micro"], tpr["micro"],  
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],  
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])  
for i, color in zip(range(n_classes-97), colors):  
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Some extension of Receiver operating characteristic to multi-class')  
plt.legend(loc="lower right")  
plt.show()


# Agrandamos la vista en la esquina superior izquierda.
plt.figure(2)  
plt.xlim(0, 0.2)  
plt.ylim(0.8, 1)  
plt.plot(fpr["micro"], tpr["micro"],  
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],  
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])  
for i, color in zip(range(3), colors):  
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Some extension of Receiver operating characteristic to multi-class')  
plt.legend(loc="lower right")  
plt.show()  

print('class for image 1: ' + str(np.argmax(y_test[0])))  
print('predicted:         ' + str(scnn_predicted[0])) 
imgplot = plt.imshow(x_train_original[0])  
plt.show()  


print('class for image 3: ' + str(np.argmax(y_test[3])))  
print('predicted:         ' + str(scnn_predicted[3])) 
imgplot = plt.imshow(x_train_original[3])  
plt.show()  


#Histórico
with open('scnn_history.txt', 'wb') as file_pi:  
  pickle.dump(scnn.history, file_pi)

# with open('simplenn_history.txt', 'rb') as f:  
#   snn_history = pickle.load(f)

# plt.figure(0)  
# plt.plot(snn_history['val_acc'],'r')  
# plt.plot(scnn.history['val_acc'],'g')  
# plt.xticks(np.arange(0, 11, 2.0))  
# plt.rcParams['figure.figsize'] = (8, 6)  
# plt.xlabel("Num of Epochs")  
# plt.ylabel("Accuracy")  
# plt.title("Simple NN Accuracy vs simple CNN Accuracy")  
# plt.legend(['simple NN','CNN'])  

# plt.figure(0)  
# plt.plot(snn_history['val_loss'],'r')  
# plt.plot(scnn.history['val_loss'],'g')  
# plt.xticks(np.arange(0, 11, 2.0))  
# plt.rcParams['figure.figsize'] = (8, 6)  
# plt.xlabel("Num of Epochs")  
# plt.ylabel("Loss")  
# plt.title("Simple NN Loss vs simple CNN Loss")  
# plt.legend(['simple NN','CNN']) 

# plt.figure(0)  
# plt.plot(snn_history['val_mean_squared_error'],'r')  
# plt.plot(scnn.history['val_mean_squared_error'],'g')  
# plt.xticks(np.arange(0, 11, 2.0))  
# plt.rcParams['figure.figsize'] = (8, 6)  
# plt.xlabel("Num of Epochs")  
# plt.ylabel("Mean Squared Error")  
# plt.title("Simple NN MSE vs simple CNN MSE")  
# plt.legend(['simple NN','CNN'])  