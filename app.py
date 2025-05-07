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
