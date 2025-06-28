## 🎨 Task-05: Neural Style Transfer

### 📝 Objective

This task demonstrates how to apply **Neural Style Transfer** to blend the **artistic style** of one image (like a painting) with the **content** of another image (like a photograph). The result is a new image that retains the structure of the content image but appears as if it were painted in the style of the style image.

---

### ⚙️ Technologies & Tools Used

* **Python**
* **TensorFlow & TensorFlow Hub**: For neural network and model handling.
* **NumPy**: For image preprocessing.
* **Matplotlib**: For displaying and saving images.
* **Pre-trained Style Transfer Model**:
  `https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2`

---

### 🔍 Features

* Uses a **pre-trained TensorFlow Hub model** for arbitrary style transfer.
* Supports **batch processing**: stylizes multiple content-style image pairs.
* Outputs are **saved automatically** to a folder named `stylized_outputs`.
* **Visualization**: Displays side-by-side comparison of content, style, and output images.

---

### 🚀 How It Works

1. **Model Loading**:
   Loads a powerful style transfer model from **TensorFlow Hub** that supports arbitrary content and style image pairs.

2. **Image Preprocessing**:

   * Loads content and style images.
   * Normalizes and reshapes them for model compatibility.
   * Style image is resized to `256x256` as required by the model.

3. **Style Transfer**:

   * Passes both images through the model.
   * Generates a new image where the content is preserved, but stylized using texture and color features from the style image.

4. **Visualization & Saving**:

   * Combines the original content, style, and stylized image into a single figure.
   * Saves the result to `stylized_outputs/<output_name>.png`.

5. **Batch Processing**:

   * The script is preconfigured to stylize 4 different content-style pairs such as:

     * `"einstien.jpg"` with `"mona-lisa.jpg"`
     * `"feyman.jpg"` with `"ggwood.jpg"`
   * Easily extendable by adding more pairs to the `image_tasks` list.

---

### 📂 Output Example

For the pair:

```python
("einstien.jpg", "mona-lisa.jpg", "einstien_monalisa")
```

The output will be:

* **Content**: Albert Einstein's photo
* **Style**: Mona Lisa painting
* **Stylized Output**: Einstein's image painted in the style of Mona Lisa
* **Saved File**: `stylized_outputs/einstien_monalisa.png`

---

### 📌 References

* [TF Hub: Arbitrary Image Stylization](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)
* [Neural Style Transfer (Wiki)](https://en.wikipedia.org/wiki/Neural_Style_Transfer)
* [Magenta Project](https://magenta.tensorflow.org/)
