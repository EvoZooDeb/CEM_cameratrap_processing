import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os
import matplotlib.pyplot as plt

img_size = 380

input_dir = "/home/golah/wolf_camtrap/training_classification_2024/standard_box_size/train/"
classes = os.listdir(input_dir)
n_classes = len(classes)
print(classes)
print(n_classes)

base_model = tf.keras.applications.EfficientNetV2L(weights = "imagenet", input_shape=(img_size, img_size, 3), include_top = False)
#base_model.summary()

base_model.trainable = False # Warm-up

# create new models
model = tf.keras.Sequential([
    base_model,
    #layers.Conv2D(1024, 3, 1, activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(2048, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1024, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation="softmax")
])

from tensorflow.keras.optimizers import Adam
adam_opt = Adam(learning_rate = 0.001)    # Warm-up
#adam_opt = Adam(learning_rate = 0.0001) # Post warm-up

# compile the model
model.compile(optimizer = adam_opt, loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall'])
#model.compile(optimizer = adam_opt, loss='categorical_focal_crossentropy', metrics=['accuracy']) # Focal-loss

train_data_aug = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    horizontal_flip = True,
    rotation_range = 0.15
#    zoom_range = 0.2,
#    vertical_flip = True
)

val_data_aug = ImageDataGenerator(
    rescale = 1./255
)

train_dir  = input_dir
val_dir    = "/home/golah/wolf_camtrap/training_classification_2024/standard_box_size/val/"

train_generator = train_data_aug.flow_from_directory(
    train_dir,
    target_size = (img_size, img_size), # Ez fölösleges ha nem mandatory arg
    batch_size  = 32,
    class_mode  = "categorical",
    color_mode  = "rgb",
    shuffle     = True
)

val_generator = val_data_aug.flow_from_directory(
    val_dir,
    target_size = (img_size, img_size), # Ez fölösleges ha nem mandatory arg
    batch_size  = 32,
    class_mode  = "categorical",
    color_mode  = "rgb",
)

epochs = 80
best_model = "./temp/4_wolf_camtrap_EfficientNetV2L_best.h5"
last_model = "./temp/4_wolf_camtrap_EfficientNetV2L_last.h5"

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

callbacks = [
        ModelCheckpoint(best_model, verbose = 1, save_best_only = True, monitor ="val_accuracy"),
        ModelCheckpoint(last_model, verbose = 1, save_best_only = False, monitor ="val_accuracy"),
        #ReduceLROnPlateau(monitor="val_accuracy", patience = 35, factor=0.1, verbose = 1, min_lr = 1e-6),
        #EarlyStopping(monitor="val_accuracy", patience = 35, verbose = 1)
]

result = model.fit(
    train_generator, epochs = epochs, validation_data = val_generator, callbacks=callbacks
)

best_val_acc_epoch = np.argmax(result.history['val_accuracy'])
best_val_acc = result.history['val_accuracy'][best_val_acc_epoch]

# Print results
print("Best val acc: " + str(best_val_acc))

# Plot acc
plt.plot(result.history["accuracy"], label = "train acc")
plt.plot(result.history["val_accuracy"], label = "val acc")
plt.legend()
plt.savefig("Fig_1.png")
plt.show()

# Plot loss
plt.plot(result.history["loss"], label = "train loss")
plt.plot(result.history["val_loss"], label = "val loss")
plt.legend()
plt.savefig("Fig_2.png")
plt.show()


