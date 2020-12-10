import matplotlib.pyplot as plt
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print('image shape: ', x_test[0].shape)

def show_test_examples(images, dim=10):
    plt.figure()

    figure, axes = plt.subplots(dim, dim)
    print(axes)
    for i, axes_row in enumerate(axes):
        for j, ax in enumerate(axes_row):
            ax.imshow(images[i * dim + j])
    plt.show()


show_test_examples(x_test)
