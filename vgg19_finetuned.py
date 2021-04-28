#Importing required packages
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from sklearn.metrics import classification_report
import keras
import tensorflow as tf 
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Dense
from keras.models import Model
import matplotlib.pyplot as plt


datapath="/bi/etl/users/anish/dataset/"
images = sorted(list(os.listdir(datapath)))


train_data=[]
train_labels=[]
test_data=[]
test_labels=[]

for imagedir in images:
    print('Loading images for '+imagedir)
    path=sorted(list(os.listdir(datapath+'/'+imagedir)))
    for img in path:
        if int(img.split('_')[1].replace('.jpg','')) in range(0,41):
            image = cv2.imread(datapath+'/'+imagedir+'/'+img)
            image = cv2.resize(image, (224,224))
            image = img_to_array(image)
            train_data.append(image)
            l = label = imagedir
            train_labels.append(l)
        else:
            image = cv2.imread(datapath+'/'+imagedir+'/'+img)
            image = cv2.resize(image, (224,224))
            image = img_to_array(image)
            test_data.append(image)
            l = label = imagedir
            test_labels.append(l)
			
print(len(train_data))
print(len(train_labels))
print(len(test_data))
print(len(test_labels))



x_train = np.array(train_data, dtype="float32") / 255.0
x_test = np.array(test_data, dtype="float32") / 255.0
labels = np.array(train_labels)
mlb = LabelBinarizer()
y_train = mlb.fit_transform(labels)
labels = np.array(test_labels)
mlb = LabelBinarizer()
y_test = mlb.fit_transform(labels)

model_VGG19 = tf.keras.models.load_model('model_vgg19')

model_VGG19.trainable = True
freezed_layers = 15
for layer in model_VGG19.layers[:freezed_layers]:
    layer.trainable =  False
	
model_VGG19.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=0.0001),
              metrics=['accuracy'])
			  
tf.config.run_functions_eagerly(True)

ftuned_vgg19 = model_VGG19.fit(x_train,y_train,epochs=10)

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(ftuned_vgg19.history["loss"])
ax.set_title("Training loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(ftuned_vgg19.history["accuracy"])
ax2.set_title("Training accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

loss, accuracy = model_VGG19.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

y_pred = model_VGG19.predict(x_test)
y_pred_encoded = tf.one_hot(tf.argmax(y_pred, axis = 1),15)
print(classification_report(y_test, y_pred_encoded))