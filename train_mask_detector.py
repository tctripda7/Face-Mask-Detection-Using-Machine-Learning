# Data Preprocessing - Converting images of datasets into arrays and performing deep learning

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4  #(0.0001) less learning rate->loss will be calculated properly=>better accuracy
EPOCHS = 20  #The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
BS = 32  #Batch Size

DIRECTORY = r"C:\Users\Tripda\Desktop\Face Mask Detection\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

data = []  #append image to array
labels = [] #label of corresponding images

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path): #images in the dataset is listed with listdir method
    	img_path = os.path.join(path, img) #join path of particular with mask image to corresponding image
    	image = load_img(img_path, target_size=(224, 224)) #coming from keras.preprocessing , converting size of the image to 224 x224
    	image = img_to_array(image)#coming from keras.preprocessing, convarting image to array
    	image = preprocess_input(image)#for mobilenet we need to use preprocess input


    	data.append(image) #numerical values,append image array to data list
    	labels.append(category) #text values, append corresponding labels

# perform one-hot encoding on the labels
lb = LabelBinarizer() #coming from sklearn preprocessing module
labels = lb.fit_transform(labels) #convert with mask and without mask data into categotrical values
labels = to_categorical(labels) #converting values to 0 and 1

# Converting data and labels into numpy array because deep learning model works with only numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42) #splitting testing and training data
# train_test_split => sklearn.model_selection
# Split arrays or matrices into random train and test subsets
# End of Preprocessing

# construct the training image generator for data augmentation
# input image processed as an array->mobilenet->max-pooling->flatten it->fully connected layer-> output
# MobileNet are faster to process in comparison to CNN and uses lesser parameters

aug = ImageDataGenerator(   #creates data augmentation i.e. creates many images with a single image by adding various properties which results in more images for training
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Modelling
# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False, #include_top =fully connected layer should be included or not
	input_tensor=Input(shape=(224, 224, 3)))  #imagenet contains pretrained image models
#3 = channels rgb

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"]) ###############

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")