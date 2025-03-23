import numpy as np
import textwrap
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

def add_headline(image_path, caption, max_width=60, display=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Unable to load the image. Please check the file path.")

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)  # Black background (opposite contrast)

    # Wrap text into multiple lines
    wrapped_text = textwrap.wrap(caption, width=max_width)

    # Calculate the box size
    line_height = 30  # Adjust spacing between lines
    text_sizes = [cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in wrapped_text]
    box_width = image.shape[1]  # Full image width
    box_height = len(wrapped_text) * line_height + 20  # Adding padding

    # Box position (bottom full-width)
    box_x = 0
    box_y = image.shape[0] - box_height  # Position at the bottom

    # Draw background rectangle
    cv2.rectangle(image, (box_x, box_y), (box_x + box_width, box_y + box_height), bg_color, -1)

    # Draw text centered inside the box
    y_start = box_y + 30  # Adjust text inside the box
    for line in wrapped_text:
        text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
        text_x = (image.shape[1] - text_size[0]) // 2  # Centered text
        cv2.putText(image, line, (text_x, y_start), font, font_scale, text_color, font_thickness)
        y_start += line_height

    # Display image if requested
    if display:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    
    return image

def florence2(image, Prompt: str = "<DETAILED_CAPTION>", Task: str = "<DETAILED_CAPTION>"):
    """
    Process an image with Florence-2 model and return the caption text only.
    
    Args:
        image: PIL Image object
        Prompt: The prompt for the model
        Task: Task type for the model
        
    Returns:
        dict: The parsed answer from the model
    """
    inputs = processor(text=Prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    parsed_answer = processor.post_process_generation(generated_text, task=Task, image_size=(image.width, image.height))
    
    # Ensure we return just the text caption
    if isinstance(parsed_answer, dict) and Prompt in parsed_answer:
        # Just return the parsed answer dictionary as before
        return parsed_answer, Prompt, Task
    else:
        # In case of unexpected structure, provide a fallback
        return {Prompt: "Unable to generate caption for this image"}, Prompt, Task

# if __name__ == "__main__":
#     image_path = 'uploads\sample2.jpg'
#     image = Image.open(image_path)
#     parsed_answer, prompt, task = florence2(image)
#     print(parsed_answer)
#     add_headline(image_path, parsed_answer[prompt])

