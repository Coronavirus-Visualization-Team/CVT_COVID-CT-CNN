from keras import layers, models, optimizers
import matplotlib.pyplot as plt
import PIL
from lung_classifier import *


def build():
    model = models.Sequential()
    # (150, 150) means the input image has to be 150 x 150 px
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(image_size[0], image_size[1], 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    return model

def graph_metrics(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def save(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_no_overfit.h5")
    print("Saved model to disk")

def train(model, train_generator, validation_generator, num_train_samples, num_val_samples):
    history = model.fit(
                    train_generator,
                    steps_per_epoch=num_train_samples // train_batch_size,
                    epochs=30,
                    validation_data=validation_generator,
                    validation_steps=num_val_samples // val_batch_size
            )
    return history

model = build()
num_train_samples, num_val_samples = initData()
train_generator, validation_generator = preprocess_data()
history = train(model, train_generator, validation_generator, num_train_samples, num_val_samples)
graph_metrics(history)
save(model)
