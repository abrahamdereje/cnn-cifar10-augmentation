import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# =============================
# STEP 1: Mount Google Drive
# =============================
# from google.colab import drive
# drive.mount('/content/drive')

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
BATCH_SIZE = 16
ROOT_DIR = "/content/drive/MyDrive/cifar10_project"
MODEL_DIR = f"{ROOT_DIR}/models"
REPORTS_DIR = f"{ROOT_DIR}/reports"
LABELS = ['airplane', 'automobile', 'bird', 'cat',
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =========================
# Load CIFAR-10
# =========================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# =========================
# Image Generators
# =========================
cnn_plain = ImageDataGenerator()
cnn_aug = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_gen_plain = cnn_plain.flow(
    x_train / 255., y_train_cat, batch_size=BATCH_SIZE)
train_gen_aug = cnn_aug.flow(
    x_train / 255., y_train_cat, batch_size=BATCH_SIZE)
val_gen = cnn_plain.flow(x_test / 255., y_test_cat,
                         batch_size=BATCH_SIZE, shuffle=False)

resnet_plain = ImageDataGenerator(preprocessing_function=preprocess_input)
resnet_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input, rotation_range=20,
    width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2,
    horizontal_flip=True
)
train_resnet_plain = resnet_plain.flow(
    x_train, y_train_cat, batch_size=BATCH_SIZE)
train_resnet_aug = resnet_aug.flow(x_train, y_train_cat, batch_size=BATCH_SIZE)
val_resnet = resnet_plain.flow(
    x_test, y_test_cat, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# Model Builders
# =========================


def build_custom_cnn(input_shape=(32, 32, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_resnet_model(trainable=True):
    base = ResNet50(include_top=False, weights='imagenet',
                    input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = trainable
    model = models.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =========================
# Train and Evaluate Function
# =========================


def train_and_evaluate(model, train_gen, val_gen, val_y, name, epochs):
    ckpt = callbacks.ModelCheckpoint(
        f"{MODEL_DIR}/{name}_best.keras", save_best_only=True)
    es = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(train_gen, validation_data=val_gen,
                        epochs=epochs, callbacks=[ckpt, es])
    model.save(f"{MODEL_DIR}/{name}.keras")

    # Accuracy & Loss Plots
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"{name} Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{name} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/{name}_metrics.png")
    plt.close()

    # Predictions
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(val_y, axis=1)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=LABELS)
    with open(f"{REPORTS_DIR}/{name}_classification_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/{name}_confusion_matrix.png")
    plt.close()


# =========================
# Train All Models
# =========================
train_and_evaluate(build_custom_cnn(), train_gen_plain,
                   val_gen, y_test_cat, "custom_no_aug", epochs=20)
train_and_evaluate(build_custom_cnn(), train_gen_aug, val_gen,
                   y_test_cat, "custom_with_aug", epochs=20)
train_and_evaluate(build_resnet_model(trainable=True), train_resnet_plain,
                   val_resnet, y_test_cat, "resnet_no_aug", epochs=15)
train_and_evaluate(build_resnet_model(trainable=True), train_resnet_aug,
                   val_resnet, y_test_cat, "resnet_with_aug", epochs=15)

# =========================
# Final Report Summary
# =========================
with open(f"{REPORTS_DIR}/model_comparison.txt", "w") as f:
    f.write("Model Comparison Results:\n")
    for name in ["custom_no_aug", "custom_with_aug", "resnet_no_aug", "resnet_with_aug"]:
        f.write(f"- {name}_metrics.png\n")
        f.write(f"- {name}_classification_report.txt\n")
        f.write(f"- {name}_confusion_matrix.png\n")

print("✅ All models trained and reports saved to:", REPORTS_DIR)
