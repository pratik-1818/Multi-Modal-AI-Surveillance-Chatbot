from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Load Florence-2 Model for Object Detection
model_id = "microsoft/Florence-2-base"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def generate_labels(image):
    """Generates object detection results."""
    task_prompt = "<OD>"  # Object Detection Task
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
