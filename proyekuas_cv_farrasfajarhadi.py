import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import kagglehub
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

path = kagglehub.dataset_download("abdelghaniaaba/wildfire-prediction-dataset")

print("Path to dataset files:", path)

tf.random.set_seed(42)

def load_and_preprocess_data(train_dir, valid_dir, test_dir, img_size=(128, 128), batch_size=32):

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        classes=['nowildfire', 'wildfire']
    )


    validation_generator = valid_test_datagen.flow_from_directory(
        valid_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        classes=['nowildfire', 'wildfire']
    )


    test_generator = valid_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        classes=['nowildfire', 'wildfire']
    )

    return train_generator, validation_generator, test_generator


def build_model(input_shape=(128, 128, 3)):
    model = models.Sequential([

        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator, epochs=15):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=4,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=1e-6
            )
        ]
    )
    return history

def evaluate_model(model, test_generator):
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig('wildfire_training_history.png')

if __name__ == "__main__":
    base_dir = path
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')
    test_dir = os.path.join(base_dir, 'test')

    img_size = (128, 128)
    batch_size = 32
    epochs = 15

    train_generator, validation_generator, test_generator = load_and_preprocess_data(
        train_dir, valid_dir, test_dir, img_size, batch_size
    )

    model = build_model(input_shape=(img_size[0], img_size[1], 3))
    model.summary()

    history = train_model(model, train_generator, validation_generator, epochs)

    evaluate_model(model, test_generator)

    plot_training_history(history)

    model.save('wildfire_classification_model.h5')

# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


# # Function to load test data and preprocess it
# def load_test_data(test_dir, img_size=(128, 128), batch_size=32):
#     test_datagen = ImageDataGenerator(rescale=1./255)
#     test_generator = test_datagen.flow_from_directory(
#         test_dir,
#         target_size=img_size,
#         batch_size=batch_size,
#         class_mode='binary',
#         shuffle=False,  # Keep order for evaluation
#         classes=['nowildfire', 'wildfire']
#     )
#     return test_generator

# # Function to make predictions and evaluate
# def predict_and_evaluate(model, test_generator):
#     # Get predictions
#     predictions = model.predict(test_generator)
#     predicted_classes = (predictions > 0.5).astype(int).flatten()  # Threshold at 0.5 for binary classification
#     true_classes = test_generator.classes
#     filenames = test_generator.filenames

#     # Calculate accuracy
#     accuracy = np.mean(predicted_classes == true_classes)
#     print(f"Prediction Accuracy on Test Set: {accuracy:.4f}")

#     return predictions, predicted_classes, true_classes, filenames

# # Function to visualize predictions
# def visualize_predictions(test_generator, predicted_classes, true_classes, filenames, num_samples=6):
#     plt.figure(figsize=(15, 10))
#     class_names = ['No Wildfire', 'Wildfire']

#     # Get a batch of images
#     images, labels = next(test_generator)
#     k = len(images)
#     for i in range(min(num_samples, k)):
#         plt.subplot(3, 10, i + 1)
#         plt.imshow(images[i])
#         true_label = class_names[int(labels[i])]
#         pred_label = class_names[predicted_classes[i]]
#         if pred_label == 1:
#           i -= 1
#         title = f"True: {true_label}\nPred: {pred_label}"
#         plt.title(title, fontsize=10)
#         plt.axis('off')

#     plt.tight_layout()
#     plt.savefig('wildfire_prediction_results.png')

# # Main execution
# if __name__ == "__main__":
#     # Paths
#     model_path = 'wildfire_classification_model.h5'
#     test_dir = os.path.join(path, 'test')  # Update with actual path from kagglehub
#     img_size = (128, 128)
#     batch_size = 32

#     # Load the trained model
#     model = tf.keras.models.load_model(model_path)

#     # Load test data
#     test_generator = load_test_data(test_dir, img_size, batch_size)

#     # Make predictions
#     predictions, predicted_classes, true_classes, filenames = predict_and_evaluate(model, test_generator)

#     # Visualize results
#     visualize_predictions(test_generator, predicted_classes, true_classes, filenames)