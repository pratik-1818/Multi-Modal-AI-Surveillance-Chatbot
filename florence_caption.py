from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Load Florence-2 Model for Captioning
model_id = "microsoft/Florence-2-base"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def generate_caption(image):
    """Generates image caption."""
    prompt = "Describe what is visible in the image."
    inputs = processor(prompt, image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(output[0], skip_special_tokens=True)
