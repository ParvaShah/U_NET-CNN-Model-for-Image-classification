# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:04:18 2018

@author: PARVA SHAH
"""


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from PIL import Image,ImageEnhance
from keras import backend as K
K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import adam
from keras.models import Model
from keras.layers import merge,core,concatenate
from keras.engine.topology import Input
#from keras.utils.np_utils import probas_to_classes as pp
#####################################


from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K


####################################



#%%

PATH = os.getcwd()
print(PATH)
# Define data path
data_path = PATH + '/xx'
data_dir_list = os.listdir(data_path)
print(data_path,data_dir_list)

img_rows=128
img_cols=128
num_channel=3
num_epoch=20

# Define the number of classes
num_classes = 2

img_data_list=[]

#Pre-Processing
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        """
        kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0

# applying different kernels to the input image
        input_img = cv2.filter2D(input_img, -1, kernel_sharpen_1)
   
        input_img = cv2.filter2D(input_img, -1, kernel_sharpen_3)

   """     
        input_img_resize=cv2.resize(input_img,(128,128))
        img_data_list.append(input_img_resize)


img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)
		
#%%
USE_SKLEARN_PREPROCESSING=False

if USE_SKLEARN_PREPROCESSING:
	# using sklearn for preprocessing
	from sklearn import preprocessing
	
	def image_to_feature_vector(image, size=(128,128)):
		# resize the image to a fixed size, then flatten the image into
		# a list of raw pixel intensities
		return cv2.resize(image, size).flatten()
	
	img_data_list=[]
	for dataset in data_dir_list:
		img_list=os.listdir(data_path+'/'+ dataset)
		print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
		for img in img_list:
			input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
			#input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
			input_img_flatten=image_to_feature_vector(input_img)
			img_data_list.append(input_img_flatten)
	
	img_data = np.array(img_data_list)
	img_data = img_data.astype('float32')
	print (img_data.shape)
	img_data_scaled = preprocessing.scale(img_data)
	print (img_data_scaled.shape)
	
	print (np.mean(img_data_scaled))
	print (np.std(img_data_scaled))
	
	print (img_data_scaled.mean(axis=0))
	print (img_data_scaled.std(axis=0))
	
	if K.image_dim_ordering()=='th':
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
		print (img_data_scaled.shape)
		
	else:
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
		print (img_data_scaled.shape)
	
	
	if K.image_dim_ordering()=='th':
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
		print (img_data_scaled.shape)
		
	else:
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
		print (img_data_scaled.shape)

if USE_SKLEARN_PREPROCESSING:
	img_data=img_data_scaled
#%%
# Assigning Labels

# Define the number of classes
num_classes = 2

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:1187]=0
labels[1187:]=1

	  
names = ['NSFW :(','Not NSFW :)']
	  
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#%%
# Defining the model

#input_shape=img_data[0].shape
#input_shape=Input((1, img_rows, img_cols))
					

########################################
   


inputs = Input(shape=(3,128,128))

conv0 = Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
conv0 = Dropout(0.2)(conv0)
conv0 = Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv0)
pool0 = MaxPooling2D((2, 2))(conv0)

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool0)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)
    #
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
conv2 = Dropout(0.2)(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)
    #
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
conv3 = Dropout(0.2)(conv3)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

up1 = UpSampling2D(size=(2, 2))(conv3)
up1 = concatenate([conv2,up1],axis=1)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
conv4 = Dropout(0.2)(conv4)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
up2 = UpSampling2D(size=(2, 2))(conv4)
up2 = concatenate([conv1,up2], axis=1)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
conv5 = Dropout(0.2)(conv5)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
up3 = UpSampling2D(size=(2, 2))(conv5)
up3 = concatenate([conv0,up3], axis=1)
conv6 = Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_first')(up3)
conv6 = Dropout(0.2)(conv6)
conv6 = Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv6)

conv7 = Conv2D(3, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv6)
conv7 = core.Reshape((3,128*128))(conv7)
#conv6 = core.Permute((3,1))(conv6)
conv7 = core.Flatten()(conv7)
#conv7 = core.Dense(64)(conv7)
#conv7 = core.Activation('relu')(conv7)
#conv7 = Dropout(0.2)(conv7)
conv7 = core.Dense(2)(conv7)

    ############
conv8 = core.Activation('softmax')(conv7)

model = Model(input=inputs, output=conv8)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])






########################################
# Viewing model_configuration

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
"""
np.shape(model.layers[0].get_weights()[0])
print(model.layers[0].get_weights()[0])
model.layers[0].trainable
"""
#%%
# Training
#hist = model.fit(X_train, y_train, batch_size=16, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))
from time import gmtime, strftime
x=strftime("%M", gmtime())
x=int(x)

checkpointer = ModelCheckpoint(filepath='_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased

model.fit(X_train, y_train, nb_epoch=num_epoch, batch_size=16, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer], validation_data=(X_test, y_test))
model.save('model.hdf5')

y=strftime("%M", gmtime())
y=int(y)
print("Total time taken in running all epochs: ")
if(y>x):
    print((y-x),"min")
else:
    print(((y+60)-x),"min")

#%%


# visualizing losses and accuracy
print(model.history.history)
train_loss=model.history.history['loss']
val_loss=model.history.history['val_loss']
train_acc=model.history.history['acc']
val_acc=model.history.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#%%

# Evaluating the model


# Testing a new image
test_image = cv2.imread('C:/Users/PARVA SHAH/Pictures/NSFW Project/110.jpg')
#test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
"""
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0

# applying different kernels to the input image
test_image = cv2.filter2D(test_image, -1, kernel_sharpen_1)
   
test_image = cv2.filter2D(test_image, -1, kernel_sharpen_3)
"""
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
   
if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
# Predicting the test image
print(("4",model.predict(test_image)))
#x=model.predict_classes(test_image)
#y_proba = model.predict(test_image)
#x = K.np_utils.probas_to_classes(y_proba)
y_prob = model.predict(test_image) 
x = y_prob.argmax(axis=-1)

if x==0 :
    print("5---NSFW :(")
else :
    print("5---Not NSFW :)")
        


#%%
# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)
#print("#################################")
#print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
#print(y_pred)
#print("#################################")
#y_pred = model.predict_classes(X_test)

y_prob = model.predict(X_test) 
y_pred = y_prob.argmax(axis=-1)

#print(y_pred)
target_names = ['class 0(NSFW)','class 3(Not NSFW)']
					
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("8--Normalized confusion matrix")
    else:
        print('8--Confusion matrix, without normalization')

    print("9",cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
plt.show()

#%%
# Saving and loading model and weights

from keras.models import load_model


"""
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
"""
"""
loaded_model=load_model('model.hdf5')
y_prob = model.predict(test_image) 
x = y_prob.argmax(axis=-1)
print("[5] ",x)
"""