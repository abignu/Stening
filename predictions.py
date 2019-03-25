#dependencies
import numpy as np  
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt  
import math  
import cv2  

top_model_weights_path = 'bottleneck_fc_model_weights.h5'

image_path = './pruebas/16.png'  

train_data_dir = '../dataset/train'

img_width, img_height = 224, 224  

datagen_top = ImageDataGenerator(rescale=1./255)

generator_top = datagen_top.flow_from_directory(  
	     train_data_dir,  
	     target_size=(img_width, img_height),  
	     batch_size=16,  
	     class_mode='categorical',  
	     shuffle=False)

orig = cv2.imread(image_path)  

print("[INFO] loading and preprocessing image...")  
image = load_img(image_path, target_size=(224, 224))  

image = img_to_array(image)  

image = cv2.resize(image,(img_width, img_height))

# important! otherwise the predictions will be '0'  
image = image / 255  

image = np.expand_dims(image, axis=0)

# build the VGG16 network  
model = applications.VGG16(include_top=False, weights='imagenet')  

# get the bottleneck prediction from the pre-trained VGG16 model  
bottleneck_prediction = model.predict(image)  

# build top model  
model = Sequential()  
model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))  
model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(6, activation='sigmoid'))  

model.load_weights(top_model_weights_path)  

# use the bottleneck prediction on the top model to get the final classification  
class_predicted = model.predict_classes(bottleneck_prediction) 

inID = class_predicted[0]  

class_dictionary = generator_top.class_indices  

inv_map = {v: k for k, v in class_dictionary.items()}  

label = inv_map[inID]  

# get the prediction label  
print("Image ID: {}, Label: {}".format(inID, label))  

# display the predictions with the image  
cv2.putText(orig, "Predicted: {}".format(label), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)  

cv2.imshow("Classification", orig)  
cv2.waitKey(0)  
cv2.destroyAllWindows()  