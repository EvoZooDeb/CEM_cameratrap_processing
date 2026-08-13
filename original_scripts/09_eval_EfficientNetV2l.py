import tensorflow as tf
import os
from keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

img_size = 380
test_dir = "/home/golah/wolf_camtrap/training_classification_2024/standard_box_size/test_2_artio/"
classes = os.listdir(test_dir)

#model_path = "/home/golah/wolf_camtrap/scripts/classification/temp/4_wolf_camtrap_EfficientNetV2L_unfrozen_best_lr00001.keras" # Combined
#model_path = "/home/golah/wolf_camtrap/scripts/classification/temp/2_wolf_camtrap_carnivora_EfficientNetV2L_unfrozen_best_lr00001.keras" # Carnivora
model_path = "/home/golah/wolf_camtrap/scripts/classification/temp/2_wolf_camtrap_artiodactyl_EfficientNetV2L_unfrozen_best_lr00001.keras" # Artio
model = tf.keras.models.load_model(model_path)

test_data_aug = ImageDataGenerator(
    rescale = 1./255
)

test_generator = test_data_aug.flow_from_directory(
        test_dir,
        target_size = (img_size, img_size),
        batch_size = 32,
        class_mode = "categorical",
        color_mode = "rgb",
)

results = model.evaluate(test_generator)
print(results)

