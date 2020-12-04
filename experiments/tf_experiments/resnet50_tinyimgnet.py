# https://www.tensorflow.org/guide/keras/preprocessing_layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create a data augmentation stage with horizontal flipping, rotations, zooms
data_augmentation = keras.Sequential(
    [
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.1),
        preprocessing.RandomZoom(0.1),
    ]
)

# Create a model that includes the augmentation stage
input_shape = (64, 64, 3)
classes = 200
inputs = keras.Input(shape=input_shape)
# Augment images
x = data_augmentation(inputs)
# Rescale image values to [0, 1]
x = preprocessing.Rescaling(1.0 / 255)(x)
# Add the rest of the model
outputs = keras.applications.ResNet50(
    weights=None, include_top=True, input_shape=input_shape, classes=classes
)(x)
model = keras.Model(inputs, outputs)

# Load some data
TINY_IMAGENET_PATH = 'datasets/tiny-imagenet-200'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_iterator = train_datagen.flow_from_directory(
    f'{TINY_IMAGENET_PATH}/train', target_size=(64, 64), batch_size=32, class_mode='binary')

for i, (x, y) in enumerate(train_iterator):
    if i == 1:
        # print(x)
        print('x', x.shape)
        # print(y)
        print('y', y[0].shape)
        probabilities = model.call(tf.convert_to_tensor(x))
        print('probs', probabilities.shape)
        break
val_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_generator.flow_from_directory(
    f'{TINY_IMAGENET_PATH}/val', target_size=(64, 64), batch_size=32, class_mode='binary')

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(
    train_iterator,
    steps_per_epoch=20,
    epochs=1,
    # validation_data=val_generator,
    # validation_steps=800
)


# x_train = x_train.reshape((len(x_train), -1))
# x_test = x_test.reshape((len(x_test), -1))
# input_shape = x_train.shape[1:]
# classes = 10

# # Create a Normalization layer and set its internal state using the training data
# normalizer = preprocessing.Normalization()
# normalizer.adapt(x_train)

# # Create a model that include the normalization layer
# inputs = keras.Input(shape=input_shape)
# x = normalizer(inputs)
# outputs = layers.Dense(classes, activation="softmax")(x)
# model = keras.Model(inputs, outputs)

# # Train the model
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# model.fit(x_train, y_train)

for x, y in train_iterator:
    print(x.shape)
    # x = x.reshape(-1,)
    probabilities = model.call(tf.convert_to_tensor(x))
    print('probs', probabilities.shape)
    predictions = tf.argmax(probabilities, 1)
    print('predictions', predictions.shape)

    count = 0
    for i in range(len(predictions)):
        if predictions[i] == y[i]:
            count += 1
        if i % 100 == 0:
            print(count, '/', i)
    print(predictions.shape)
    # y_test = y_test.reshape(len(y_test))
    print(y.shape)
    print(tf.reduce_mean(tf.cast(predictions == y, tf.float32)))
