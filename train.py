from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

dataset = #Dataset Directory

init_learning_rate = 1e-4 
epochs = 20
batch_size = 32 

image_paths = list(paths.list_images(dataset))
data = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

image_data_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

input_model = MobileNetV2(
        weights="imagenet", 
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

output_model = input_model.output
output_model = Conv2D(32, (5, 5), padding="same", activation="relu")(output_model)
output_model = AveragePooling2D(pool_size=(5, 5), strides=1, padding="same")(output_model)
output_model = Flatten(name="flatten")(output_model)
output_model = Dense(32, activation="relu")(output_model)
output_model = Dense(64, activation="relu")(output_model)
output_model = Dropout(0.5)(output_model)
output_model = Dense(32, activation="relu")(output_model)
output_model = Dense(2, activation="softmax")(output_model)

model = Model(inputs=input_model.input, outputs=output_model)

for layer in input_model.layers:
    layer.trainable = False

optimizer = Nadam(learning_rate=init_learning_rate, decay=init_learning_rate / epochs)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
train = model.fit(
        image_data_generator.flow(trainX, trainY, batch_size=batch_size),
        steps_per_epoch=len(trainX) // batch_size,
        validation_data=(testX, testY),
        validation_steps=len(testX) // batch_size,
        epochs=epochs)

predict = model.predict(testX, batch_size=batch_size)
predict_index = np.argmax(predict, axis=1)
print(classification_report(testY.argmax(axis=1), predict_index, target_names=label_binarizer.classes_))

model.save(#Model Save Directory)