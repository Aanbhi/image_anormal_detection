from transformers import Owlv2Processor, Owlv2ForObjectDetection
import cv2
import torch
from PIL import Image
import numpy as np
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import matplotlib.pyplot as plt

# Initialize the processor and model
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", cache_dir="/path/to/local/cache")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", cache_dir="/path/to/local/cache")

# Function to process a single frame and return the frame with bounding boxes
def OWLv2_process_frame(frame, texts):
    # Convert frame to PIL image
    image = Image.fromarray(frame)
    
    # Process the image
    inputs = processor(text=texts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)
    
    # Retrieve predictions for the first image for the corresponding text queries
    i = 0
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    
    # Draw bounding boxes on the frame
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"{texts[label]}: {score:.3f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame


# if __name__ == "__main__":
#     # Define the texts to detect
#     texts = [['bag', 'child', 'person']]
#     image = r"sample2.jpg"
#     Frame = cv2.imread(image)
#     frame = OWLv2_process_frame(Frame, texts[0])
    
#     plt.imshow(frame)
#     plt.show()