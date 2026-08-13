import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os
import matplotlib.pyplot as plt

img_size = 380

input_dir = "/home/golah/wolf_camtrap/training_classification_2024/standard_box_size/train_artio/"
classes = os.listdir(input_dir)
n_classes = len(classes)
print(classes)
print(n_classes)

model =  tf.keras.models.load_model("/home/golah/wolf_camtrap/scripts/classification/temp/2_wolf_camtrap_artiodactyl_EfficientNetV2L_epoch80.h5")
#model =  tf.keras.models.load_model("/home/golah/wolf_camtrap/scripts/classification/temp/1_wolf_camtrap_carnivora_EfficientNetV2L_epoch80.h5")
type(model)
model.summary()
for layer in model.layers:
    layer.trainable = True

from tensorflow.keras.optimizers import Adam
adam_opt = Adam(learning_rate = 0.00001)    # Warm-up
#adam_opt = Adam(learning_rate = 0.0001) # Post warm-up

# compile the model
model.compile(optimizer = adam_opt, loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall'])
#model.compile(optimizer = adam_opt, loss='categorical_focal_crossentropy', metrics=['accuracy']) # Focal-loss

train_data_aug = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    horizontal_flip = True,
    rotation_range = 0.15
    #zoom_range = 0.2,
    #vertical_flip = True
)

val_data_aug = ImageDataGenerator(
    rescale = 1./255
)

train_dir  = input_dir
val_dir    = "/home/golah/wolf_camtrap/training_classification_2024/standard_box_size/val_artio"

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

epochs = 40
last_model = "./temp/2_wolf_camtrap_artiodactyl_EfficientNetV2L_unfrozen_last_lr00001.keras"
best_model = "./temp/2_wolf_camtrap_artiodactyl_EfficientNetV2L_unfrozen_best_lr00001.keras"

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

callbacks = [
        ModelCheckpoint(last_model, verbose = 1, save_best_only = False, monitor ="val_accuracy"),
        ModelCheckpoint(best_model, verbose = 1, save_best_only = True, monitor ="val_accuracy"),
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


