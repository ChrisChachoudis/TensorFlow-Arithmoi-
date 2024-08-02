import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Upload του dataset MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize των εικόνων
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create τo μοντέλου
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Σύνταξη του μοντέλου
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training του μοντέλου
model.fit(train_images, train_labels, epochs=5)

# Αξιολόγηση του μοντέλου
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'\nΑκρίβεια στο test set: {test_acc}')

# Προβολή μερικών προβλέψεων
predictions = model.predict(test_images)

# Preview των πρώτων 5 εικόνων και των προβλέψεών τους
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f'Πραγματική ετικέτα: {test_labels[i]}')
    plt.title(f'Προβλεπόμενη ετικέτα: {predictions[i].argmax()}')
    plt.show()