import keras_deep_learning.projects.pokemon_classifier as pc
import keras_deep_learning.projects.pokemon_classifier.dataset as ds
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras_deep_learning.plotting import plot_acc_and_loss
from math import ceil


base_dir = pc.__file__.replace('__init__.py', 'data\\')
model_dir = base_dir.replace('data\\', 'models\\')
train_dir = base_dir + 'train\\'
val_dir = base_dir + 'validation\\'
test_dir = base_dir + 'test\\'

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, 
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.1,
    zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

b_size = 8
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=b_size,
    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=b_size,
    class_mode='categorical')


def get_trained_model():
    model = models.Sequential()
    conv_base = VGG16(weights='imagenet', include_top=False,
        input_shape=(150, 150, 3))
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(149, activation='softmax'))
    
    conv_base.trainable = False

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
        loss='binary_crossentropy', metrics=['acc'])

    train_steps = ceil(106 / b_size)
    val_steps = ceil(26 / b_size)

    # first training
    history = model.fit_generator(
        train_generator, steps_per_epoch=train_steps, epochs=20,
        validation_data=validation_generator, validation_steps=val_steps)
    
    for layer in conv_base.layers:
        if 'block5' in layer.name or 'block4' in layer.name:
            layer.trainable = True
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
        loss='binary_crossentropy', metrics=['acc'])
    # fine tuning
    history = model.fit_generator(
        train_generator, steps_per_epoch=train_steps, epochs=40,
        validation_data=validation_generator, validation_steps=val_steps)
    
    return model, history


model, history = get_trained_model()
plot_acc_and_loss(history.history, acc_str=True)
model.save(model_dir + 'full_classifier_pretrained.h5')