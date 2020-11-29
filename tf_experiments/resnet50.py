# https://www.tensorflow.org/guide/keras/preprocessing_layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers

# Create a data augmentation stage with horizontal flipping, rotations, zooms
data_augmentation = keras.Sequential(
    [
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.1),
        preprocessing.RandomZoom(0.1),
    ]
)

# Create a model that includes the augmentation stage
input_shape = (32, 32, 3)
classes = 10
inputs = keras.Input(shape=input_shape)
# Augment images
x = data_augmentation(inputs)
# Rescale image values to [0, 1]
x = preprocessing.Rescaling(1.0 / 255)(x)
# Add the rest of the model
outputs = keras.applications.ResNet50(
    weights=None, input_shape=input_shape, classes=classes
)(x)
model = keras.Model(inputs, outputs)

# Load some data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))
input_shape = x_train.shape[1:]
classes = 10

# Create a Normalization layer and set its internal state using the training data
normalizer = preprocessing.Normalization()
normalizer.adapt(x_train)

# Create a model that include the normalization layer
inputs = keras.Input(shape=input_shape)
x = normalizer(inputs)
outputs = layers.Dense(classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

# Train the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(x_train, y_train)

print(x_test.shape)
probabilities = model.call(tf.convert_to_tensor(x_test))
print('probs', probabilities)
predictions = tf.argmax(probabilities, 1)
print('predictions', predictions)
print('y_test', y_test)
count = 0
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        count += 1
    if i % 100 == 0:
        print(count, '/', i)
print(predictions.shape)
y_test = y_test.reshape(len(y_test))
print(y_test.shape)
print(tf.reduce_mean(tf.cast(predictions == y_test, tf.float32)))
