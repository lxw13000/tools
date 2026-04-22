---
license: apache-2.0
pipeline_tag: image-classification
---
# Model Card: Fine-Tuned Vision Transformer (ViT) for NSFW Image Classification

## Model Description

The **Fine-Tuned Vision Transformer (ViT)** is a variant of the transformer encoder architecture, similar to BERT, that has been adapted for image classification tasks. This specific model, named "google/vit-base-patch16-224-in21k," is pre-trained on a substantial collection of images in a supervised manner, leveraging the ImageNet-21k dataset. The images in the pre-training dataset are resized to a resolution of 224x224 pixels, making it suitable for a wide range of image recognition tasks.

During the training phase, meticulous attention was given to hyperparameter settings to ensure optimal model performance. The model was fine-tuned with a judiciously chosen batch size of 16. This choice not only balanced computational efficiency but also allowed for the model to effectively process and learn from a diverse array of images.

To facilitate this fine-tuning process, a learning rate of 5e-5 was employed. The learning rate serves as a critical tuning parameter that dictates the magnitude of adjustments made to the model's parameters during training. In this case, a learning rate of 5e-5 was selected to strike a harmonious balance between rapid convergence and steady optimization, resulting in a model that not only learns swiftly but also steadily refines its capabilities throughout the training process.

This training phase was executed using a proprietary dataset containing an extensive collection of 80,000 images, each characterized by a substantial degree of variability. The dataset was thoughtfully curated to include two distinct classes, namely "normal" and "nsfw." This diversity allowed the model to grasp nuanced visual patterns, equipping it with the competence to accurately differentiate between safe and explicit content.

The overarching objective of this meticulous training process was to impart the model with a deep understanding of visual cues, ensuring its robustness and competence in tackling the specific task of NSFW image classification. The result is a model that stands ready to contribute significantly to content safety and moderation, all while maintaining the highest standards of accuracy and reliability.
## Intended Uses & Limitations

### Intended Uses
- **NSFW Image Classification**: The primary intended use of this model is for the classification of NSFW (Not Safe for Work) images. It has been fine-tuned for this purpose, making it suitable for filtering explicit or inappropriate content in various applications.

### How to use
Here is how to use this model to classifiy an image based on 1 of 2 classes (normal,nsfw):

```markdown

# Use a pipeline as a high-level helper
from PIL import Image
from transformers import pipeline 

img = Image.open("<path_to_image_file>")
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
classifier(img)

```

<hr>

``` markdown

# Load model directly
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor

img = Image.open("<path_to_image_file>")
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
processor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')
with torch.no_grad():
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

predicted_label = logits.argmax(-1).item()
model.config.id2label[predicted_label]

```

<hr>
Run Yolo Version

``` markdown

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import onnxruntime as ort
import json # Added import for json

# Predict using YOLOv9 model
def predict_with_yolov9(image_path, model_path, labels_path, input_size):
    """
    Run inference using the converted YOLOv9 model on a single image.

    Args:
        image_path (str): Path to the input image file.
        model_path (str): Path to the ONNX model file.
        labels_path (str): Path to the JSON file containing class labels.
        input_size (tuple): The expected input size (height, width) for the model.

    Returns:
        str: The predicted class label.
        PIL.Image.Image: The original loaded image.
    """
    def load_json(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    # Load labels
    labels = load_json(labels_path)

    # Preprocess image
    original_image = Image.open(image_path).convert("RGB")
    image_resized = original_image.resize(input_size, Image.Resampling.BILINEAR)
    image_np = np.array(image_resized, dtype=np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))  # [C, H, W]
    input_tensor = np.expand_dims(image_np, axis=0).astype(np.float32)

    # Load YOLOv9 model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name # Assuming classification output

    # Run inference
    outputs = session.run([output_name], {input_name: input_tensor})
    predictions = outputs[0]

    # Postprocess predictions (assuming classification output)
    # Adapt this section if your model output is different (e.g., detection boxes)
    predicted_index = np.argmax(predictions)
    predicted_label = labels[str(predicted_index)] # Assumes labels are indexed by string numbers

    return predicted_label, original_image

# Display prediction for a single image
def display_single_prediction(image_path, model_path, labels_path, input_size):
    """
    Predicts the class for a single image and displays the image with its prediction.

    Args:
        image_path (str): Path to the input image file.
        model_path (str): Path to the ONNX model file.
        labels_path (str): Path to the JSON file containing class labels.
        input_size (tuple): The expected input size (height, width) for the model.
    """
    try:
        # Run prediction
        prediction, img = predict_with_yolov9(image_path, model_path, labels_path, input_size)

        # Display image and prediction
        fig, ax = plt.subplots(1, 1, figsize=(8, 8)) # Create a single plot
        ax.imshow(img)
        ax.set_title(f"Prediction: {prediction}", fontsize=14)
        ax.axis("off") # Hide axes ticks and labels

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- Main Execution ---

# Paths and parameters - **MODIFY THESE**
single_image_path = "path/to/your/single_image.jpg"  # <--- Replace with the actual path to your image file
model_path = "path/to/your/yolov9_model.onnx"    # <--- Replace with the actual path to your ONNX model
labels_path = "path/to/your/labels.json"        # <--- Replace with the actual path to your labels JSON file
input_size = (224, 224)                         # Standard input size, adjust if your model differs

# Check if the image file exists before proceeding (optional but recommended)
if os.path.exists(single_image_path):
    # Run prediction and display for the single image
    display_single_prediction(single_image_path, model_path, labels_path, input_size)
else:
    print(f"Error: The specified image file does not exist: {single_image_path}")

```

<hr>



### Limitations
- **Specialized Task Fine-Tuning**: While the model is adept at NSFW image classification, its performance may vary when applied to other tasks.
- Users interested in employing this model for different tasks should explore fine-tuned versions available in the model hub for optimal results.

## Training Data

The model's training data includes a proprietary dataset comprising approximately 80,000 images. This dataset encompasses a significant amount of variability and consists of two distinct classes: "normal" and "nsfw." The training process on this data aimed to equip the model with the ability to distinguish between safe and explicit content effectively.

### Training Stats
``` markdown

- 'eval_loss': 0.07463177293539047,
- 'eval_accuracy': 0.980375, 
- 'eval_runtime': 304.9846, 
- 'eval_samples_per_second': 52.462, 
- 'eval_steps_per_second': 3.279

```

<hr>


**Note:** It's essential to use this model responsibly and ethically, adhering to content guidelines and applicable regulations when implementing it in real-world applications, particularly those involving potentially sensitive content.

For more details on model fine-tuning and usage, please refer to the model's documentation and the model hub.

## References

- [Hugging Face Model Hub](https://huggingface.co/models)
- [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929)
- [ImageNet-21k Dataset](http://www.image-net.org/)

**Disclaimer:** The model's performance may be influenced by the quality and representativeness of the data it was fine-tuned on. Users are encouraged to assess the model's suitability for their specific applications and datasets.