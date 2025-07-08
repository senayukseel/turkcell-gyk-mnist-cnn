import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os 

model = tf.keras.models.load_model("model.keras")

data_dir = "data"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True) 

image_files = [f for f in os.listdir(data_dir) if f.endswith("-image.png")]
image_files.sort() 

for img_file in image_files:
    img_path = os.path.join(data_dir, img_file)
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original_display = original.copy()

    resized = cv2.resize(original, (28, 28))
    resized_norm = resized / 255.0
    input_image = resized_norm.reshape(1, 28, 28, 1)

    predictions = model.predict(input_image)[0]
    predicted_digit = np.argmax(predictions)

    print(f"\nGörsel: {img_file}")
    print("Rakamların Güven Skoru:")
    for i, score in enumerate(predictions):
        print(f"Rakam {i}: {score * 100:.2f}%")
    print(f"Tahmin Edilen Rakam: {predicted_digit}")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # orijinal resim 
    axs[0].imshow(original_display, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # 28x28 yeniden boyutlandırılmış hali
    axs[1].imshow(resized_norm, cmap='gray')
    axs[1].set_title("Resize Image")
    axs[1].axis('off')

    # güven skoru grafiği
    axs[2].bar(range(10), predictions * 100)
    axs[2].set_xticks(range(10))
    axs[2].set_xlabel("Rakam")
    axs[2].set_ylabel("Güven Oranı (%)")
    axs[2].set_title(f"Predict Result: {predicted_digit}")

    output_path = os.path.join(output_dir, img_file.replace("-image.png", "-output.png"))
    plt.tight_layout()
    plt.savefig(output_path)  
    plt.show()              
    plt.close(fig)            
print("Tüm çıktılar outputs klasörüne kaydedildi.")
