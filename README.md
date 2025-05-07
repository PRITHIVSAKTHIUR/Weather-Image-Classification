![1.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/DLSG05GqVrEJR7dE3VoiV.png)

# Weather-Image-Classification

> Weather-Image-Classification is a vision-language model fine-tuned from google/siglip2-base-patch16-224 for multi-class image classification. It is trained to recognize weather conditions from images using the SiglipForImageClassification architecture.

```py
Classification Report:
                 precision    recall  f1-score   support

cloudy/overcast     0.8493    0.8762    0.8625      6702
     foggy/hazy     0.8340    0.8128    0.8233      1261
     rain/strom     0.7644    0.7592    0.7618      1927
    snow/frosty     0.8341    0.8448    0.8394      1875
      sun/clear     0.9124    0.8846    0.8983      6274

       accuracy                         0.8589     18039
      macro avg     0.8388    0.8355    0.8371     18039
   weighted avg     0.8595    0.8589    0.8591     18039
```

![download (1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/T3MuycHMZDoAhjp3V5Z0p.png)

---

## Label Space: 5 Classes

The model classifies an image into one of the following weather categories:

```json
"id2label": {
  "0": "cloudy/overcast",
  "1": "foggy/hazy",
  "2": "rain/storm",
  "3": "snow/frosty",
  "4": "sun/clear"
}
```

---

## Install Dependencies

```bash
pip install -q transformers torch pillow gradio
```

---

## Inference Code

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Weather-Image-Classification"  # Replace with actual path
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {
    "0": "cloudy/overcast",
    "1": "foggy/hazy",
    "2": "rain/storm",
    "3": "snow/frosty",
    "4": "sun/clear"
}

def classify_weather(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_weather,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=5, label="Weather Condition"),
    title="Weather-Image-Classification",
    description="Upload an image to identify the weather condition (sun, rain, snow, fog, or clouds)."
)

if __name__ == "__main__":
    iface.launch()
```

---

## Intended Use

Weather-Image-Classification is useful for:

* Automated weather tagging for photography and media.
* Enhancing dataset labeling in weather-related research.
* Supporting smart surveillance and traffic systems.
* Improving scene understanding in autonomous vehicles. 
