import keras_deep_learning.projects.pokemon_classifier as pc
import keras_deep_learning.projects.pokemon_classifier.dataset as ds
from keras import layers, models, optimizers
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

b_size = 1
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(250, 250), batch_size=b_size,
    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
    val_dir, target_size=(250, 250), batch_size=b_size,
    class_mode='categorical')
#test_generator = test_datagen.flow_from_directory(
#   val_dir, target_size=(250, 250), batch_size=10)

def get_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
        input_shape=(250, 250, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
        optimizer=optimizers.RMSprop(learning_rate=1e-4),
        metrics=['acc'])
    return model


def train_model(md):
    train_steps = ceil(106 / b_size)
    val_steps = ceil(26 / b_size)
    history = md.fit_generator(
        train_generator, steps_per_epoch=train_steps, epochs=60,
        validation_data=validation_generator, validation_steps=val_steps)
    return history


if __name__ == '__main__':
    model = get_model()
    history = train_model(model)
    #model = models.load_model(model_dir + 'full_classifier_revised_data.h5')
    #loss, acc = model.evaluate_generator(test_generator, ceil(ds.get_num_val_samples() / b_size))
    #print('Model achieved %s%% accuracy on the test set' % round(acc * 100, 1))
    #model.save(model_dir + 'full_classifier_revised_data_batch1.h5')

    plot_acc_and_loss(history.history, acc_str=True)