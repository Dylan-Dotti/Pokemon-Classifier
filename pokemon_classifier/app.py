import keras_deep_learning.projects.pokemon_classifier as pc
import numpy as np
from keras import models
from keras.preprocessing.image import load_img, img_to_array


model_path = pc.__file__.replace('__init__.py',
    'models\\5_classifier.h5')
model = models.load_model(model_path)

img = load_img('C:\\Users\\Dylan\\Pictures\\mewtwo.png',
    target_size=(150, 150))
img_arr = img_to_array(img)
img_tensor = np.array([img_arr])
results = model.predict(img_tensor)[0].tolist()
res_index = results.index(max(results))

index_class_map = {
    0: 'Bulbasaur', 1: 'Charmander',
    2: 'Mewtwo', 3: 'Pikachu', 4: 'Squirtle'
}

print(index_class_map[res_index])