import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report
from collections import defaultdict

# =====================
# CONFIG
# =====================
ROOT_DIR = "/content/drive/MyDrive/cifar10_project"
MODEL_DIR = f"{ROOT_DIR}/models"
REPORTS_DIR = f"{ROOT_DIR}/reports"
TEST_DIR = f"{ROOT_DIR}/test"
LABELS = ['airplane', 'automobile', 'bird', 'cat',
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MODELS = [
    ("custom_no_aug.keras", False),
    ("custom_with_aug.keras", False),
    ("resnet_no_aug.keras", True),
    ("resnet_with_aug.keras", True),
]

# =====================
# Helper: Load images
# =====================


def load_images(test_dir, img_size, use_preprocess=False):
    x, y = [], []
    for label_idx, class_name in enumerate(LABELS):
        class_dir = os.path.join(test_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(class_dir, fname)
                img = load_img(fpath, target_size=(img_size, img_size))
                arr = img_to_array(img)
                if use_preprocess:
                    arr = preprocess_input(arr)
                else:
                    arr = arr / 255.0
                x.append(arr)
                y.append(label_idx)
    return np.array(x), np.array(y)

# =====================
# Evaluation Function
# =====================


def evaluate_model(model_name, is_resnet):
    print(f"\n🔍 Evaluating model: {model_name}")
    model_path = os.path.join(MODEL_DIR, model_name)
    model = load_model(model_path)

    img_size = 224 if is_resnet else 32
    preprocess = is_resnet

    x_test, y_test = load_images(TEST_DIR, img_size, use_preprocess=preprocess)

    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)

    correct = np.sum(y_pred == y_test)
    overall_acc = correct / len(y_test)

    # Per-class accuracy
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    for t, p in zip(y_test, y_pred):
        per_class_total[t] += 1
        if t == p:
            per_class_correct[t] += 1

    result_str = f"\n=============================\n📦 Model: {model_name.replace('.keras', '')}\n"
    result_str += f"✅ Overall Accuracy: {overall_acc:.2f} ({correct} / {len(y_test)})\n📊 Per-class Accuracy:\n"
    for i, label in enumerate(LABELS):
        acc = per_class_correct[i] / \
            per_class_total[i] if per_class_total[i] > 0 else 0
        result_str += f"  - {label}: {acc:.2f}\n"

    return result_str


# =====================
# Run Evaluation
# =====================
print("📂 Scanning test images...")
results = ""
for model_name, is_resnet in MODELS:
    results += evaluate_model(model_name, is_resnet)

# Save report
os.makedirs(REPORTS_DIR, exist_ok=True)
with open(os.path.join(REPORTS_DIR, "evaluation_report.txt"), "w") as f:
    f.write(results)

print("\n✅ Evaluation complete. Report saved to:")
print(os.path.join(REPORTS_DIR, "evaluation_report.txt"))
