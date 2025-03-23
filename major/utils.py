import os
import cv2
import numpy as np
from PIL import Image
import io
import time
from fastapi import UploadFile
import matplotlib.pyplot as plt
from models.florence2 import florence2, add_headline
from models.OWLv2 import OWLv2_process_frame
from models.yolov11 import YOLO_process_image, model as yolo_model
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

async def save_uploaded_file(file: UploadFile) -> str:
    """Save an uploaded file to disk and return its path"""
    # Create a unique filename with timestamp to avoid caching issues
    timestamp = int(time.time())
    
    # Clean the filename - replace spaces with underscores
    clean_filename = file.filename.replace(" ", "_")
    filename = f"{timestamp}_{clean_filename}"
    file_location = f"uploads/{filename}"
    
    # Write file to disk
    with open(file_location, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return file_location

def save_processed_image(image, original_path, model_name):
    """Save a processed image to disk and return its path"""
    # Create a unique filename based on the original and model
    base_filename = os.path.basename(original_path)
    timestamp = int(time.time())
    
    # Clean the filename - replace spaces with underscores
    clean_filename = base_filename.replace(" ", "_")
    filename = f"{timestamp}_{model_name}_{clean_filename}"
    file_location = f"processed/{filename}"
    
    # Ensure the directory exists
    os.makedirs("processed", exist_ok=True)
    
    # Make sure we're saving as RGB for proper web display
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to BGR if it's in RGB (OpenCV uses BGR)
        if image[0, 0, 0] > image[0, 0, 2]:  # Simple check for RGB vs BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save the image
    cv2.imwrite(file_location, image)
    
    return file_location

async def process_image(file_path: str, model_name: str):
    """Process an image with the specified model"""
    try:
        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to load image from {file_path}")
        
        processed_image = None
        result_text = "No processing applied"
        
        if model_name == "florence2":
            # Florence-2 captioning
            try:
                pil_image = Image.open(file_path)
                result, prompt, task = florence2(pil_image)
                
                # Get the caption
                caption = result[prompt]
                
                # Add the caption to the image
                processed_image = add_headline(file_path, caption, display=False)
                
                # Convert BGR to RGB for display
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                result_text = caption
            except Exception as e:
                print(f"Error in Florence2 processing: {str(e)}")
                # Return original image with error message
                processed_image = image.copy()
                cv2.putText(
                    processed_image, 
                    f"Florence2 error: {str(e)}", 
                    (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2, 
                    cv2.LINE_AA
                )
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                result_text = f"Error: {str(e)}"
            
        elif model_name == "owlv2":
            # OWLv2 object detection
            try:
                texts = ['bag', 'child', 'person']
                processed_image = OWLv2_process_frame(image.copy(), texts)
                result_text = f"OWLv2 detection of: {', '.join(texts)}"
            except Exception as e:
                print(f"Error in OWLv2 processing: {str(e)}")
                processed_image = image.copy()
                cv2.putText(
                    processed_image, 
                    f"OWLv2 error: {str(e)}", 
                    (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2, 
                    cv2.LINE_AA
                )
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                result_text = f"Error: {str(e)}"
            
        elif model_name == "yolov11":
            # YOLOv11 detection
            try:
                processed_image = YOLO_process_image(image.copy(), yolo_model)
                result_text = "YOLOv11 Fighting Detection Results"
            except Exception as e:
                print(f"Error in YOLOv11 processing: {str(e)}")
                processed_image = image.copy()
                cv2.putText(
                    processed_image, 
                    f"YOLOv11 error: {str(e)}", 
                    (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2, 
                    cv2.LINE_AA
                )
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                result_text = f"Error: {str(e)}"
        
        # If no specific processing was done, just return the original image
        if processed_image is None:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert BGR to RGB if not already done
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            # Check if we need to convert from BGR to RGB
            if np.array_equal(processed_image[0, 0], image[0, 0]):
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # Save the processed image and return its path
        processed_path = save_processed_image(processed_image, file_path, model_name)
        
        return processed_path, result_text
    except Exception as e:
        # Create a blank image with error message if something goes wrong
        print(f"General processing error: {str(e)}")
        blank_image = np.zeros((400, 600, 3), np.uint8)
        cv2.putText(
            blank_image, 
            f"Error: {str(e)}", 
            (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2, 
            cv2.LINE_AA
        )
        
        # Save the error image
        error_path = save_processed_image(blank_image, file_path, f"error_{model_name}")
        
        return error_path, f"Error: {str(e)}"