import pathlib  # Библиотека для работы с директориями
from matplotlib import pyplot as plt  # Библиотека для графиков
import numpy as np  # Библиотека для вычислений

# Библиотека для нейронной сети
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential

IMG_WIDTH, IMG_HEIGHT = 180, 180


def load_and_preprocess_images(path, batch_size):
    # Указываем путь до папки с датасетом
    dataset_dir = pathlib.Path(path)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(f"Class names: {class_names}")
    return train_ds, val_ds, class_names


# Кэшируем изображения
def cash_images(train_ds, val_ds):
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds


# Создаем модель
def create_model(class_names):
    num_classes = len(class_names)

    model = Sequential([
        layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.Rescaling(1. / 255),

        # аугментация
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),

        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        # регуляризация
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    return model


# Компилируем модель
def compile_model(model):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return


# Выводим вычисления
def print_model_summary(model):
    return model.summary()


# Тренируем модель
def train_model(model, train_ds, val_ds, epochs):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Отображаем процесс обучения
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    return


# Сохраняем веса
def save_model_weights(model):
    model.save_weights('my_model.weights.h5')
    print(f"Saved model to disk")
    return


# Загружаем веса
def load_model_weights(model):
    model.load_weights('my_model.weights.h5')
    return


# Получаем веса
def get_model_weights(model, train_ds):
    loss, acc = model.evaluate(train_ds, verbose=2)
    print(f'Restored model, accuracy: {acc * 100:5.2f}%')
    return


# Загружаем тестовое изображение и возвращаем его в виде массива NumPy
def load_test_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img, img_array


# Вычисляем вероятность
def predict_model(model, img_array):
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return score


# Выводим результат
def output_result(class_names, score, img):
    print(f"Predicted: {class_names[np.argmax(score)]} {np.max(score) * 100:5.2f}%")
    plt.imshow(img, cmap='Accent')
    plt.show()
    return


# Загружаем датасет, создаем модель и тренируем
def create_and_train_model(path, batch_size, epochs):
    # Загружаем и подготавливаем датасет
    train_ds, val_ds, class_names = load_and_preprocess_images(path, batch_size)
    train_ds, val_ds = cash_images(train_ds, val_ds)

    # Создаем модель и тренируем
    model = create_model(class_names)
    compile_model(model)
    print_model_summary(model)
    train_model(model, train_ds, val_ds, epochs)

    # Получаем веса
    get_model_weights(model, train_ds)
    return model, class_names


# Загружаем изображение и получаем предсказание
def get_image_score(model, path):
    # Загружаем изображение и преобразуем его в массив
    img, image_array = load_test_image(path)

    # Вычисляем вероятность
    score = predict_model(model, image_array)
    return img, score


def main():
    batch_size = 64
    epochs = 20
    dataset_path = 'C:/Users/SashaAlina/PythonProjects/myproject/static/images/mashina_vid_sboku'
    image_path = 'C:/Users/SashaAlina/PythonProjects/myproject/static/images/mashina_vid_sboku/test.jpg'
    model, class_names = create_and_train_model(dataset_path, batch_size, epochs)
    image, score = get_image_score(model, image_path)
    output_result(class_names, score, image)
    return


if __name__ == '__main__':
    main()
