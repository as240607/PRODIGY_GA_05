import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

# Setup
print("TF Version: ", tf.__version__)
print("TF Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# Create an output directory
os.makedirs("stylized_outputs", exist_ok=True)

# Load TF Hub model
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image(image_path):
    image = plt.imread(image_path).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def stylize_image(content_path, style_path, output_name):
    # Load original images
    content_img = plt.imread(content_path)
    style_img = plt.imread(style_path)
    
    # Preprocess for model
    content = load_image(content_path)
    style = tf.image.resize(load_image(style_path), (256, 256))

    # Generate stylized image
    outputs = hub_module(tf.constant(content), tf.constant(style))
    stylized = outputs[0].numpy().squeeze()

    # Plot all three images
    plt.figure(figsize=(20, 10))
    
    for i, (img, title) in enumerate(zip([content_img, style_img, stylized],
                                         ["Content Image", "Style Image", "Stylized Output"])):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    # Save result
    output_path = os.path.join("stylized_outputs", f"{output_name}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# List of content and style image pairs
image_tasks = [
    ("einstien.jpg", "mona-lisa.jpg", "einstien_monalisa"),
    ("feyman.jpg", "mona-lisa.jpg", "feyman_monalisa"),
    ("feyman.jpg", "ggwood.jpg", "feyman_ggwood"),
    ("einstien.jpg", "ggwood.jpg", "einstien_ggwood")
]

# Run stylization for each image pair
for content_path, style_path, output_name in image_tasks:
    stylize_image(content_path, style_path, output_name)
