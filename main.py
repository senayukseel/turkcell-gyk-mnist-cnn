import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("model.keras")

image_folder = "data/"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files = image_files[:]

images, file_names, true_labels = [], [], []

for filename in image_files:
    path = os.path.join(image_folder, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Görsel okunamadı!: {filename}")
        continue

    img_resized = cv2.resize(img, (28, 28))
    images.append(np.expand_dims(img_resized / 255.0, axis=-1))
    file_names.append(filename)

    try:
        label = int(filename.split('_')[0])
    except ValueError:
        print(f"Dosya ismi ayrıştırılamadı!: {filename}")
        continue
    true_labels.append(label)

images_np = np.array(images) 
predictions = model.predict(images_np)
predicted_classes = np.argmax(predictions, axis=1)

# Doğruluk hesapla
accuracy = np.mean(np.array(true_labels) == predicted_classes)
print(f"Doğruluk: {accuracy:.2%}")

df = pd.DataFrame({
    "Dosya Adı": file_names,
    "Gerçek Etiket": true_labels,
    "Tahmin Edilen": predicted_classes
})

df.to_csv("tahmin_sonuclari.csv", index=False, encoding="utf-8")
print("CSV dosyası kaydedildi.")

# Görselleştirme
plt.figure(figsize=(10, 6))
for i in range(len(images_np)):
    plt.subplot(2, 5, i+1)
    plt.imshow(images_np[i].reshape(28, 28), cmap="gray")
    plt.title(f"{file_names[i]}\nTahmin: {predicted_classes[i]}")
    plt.axis("off")
plt.suptitle("Görsellerin Tahminleri", fontsize=16)
plt.show()







