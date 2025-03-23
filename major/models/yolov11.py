import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLOv11 model
model = YOLO("yolov11-fighting/runs/kaggle/working/runs/detect/train/weights/best.pt", task="detect")


def YOLO_process_image(frame, model):
    # Read the image
    
    # Run inference on the frame
    results = model(frame)
    
    # Process each detection
    for idx, result in enumerate(results[0].boxes):
        # Get box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        
        # Get confidence and class
        confidence = float(result.conf[0])
        class_id = int(result.cls[0])
        
        if confidence > 0.3:  # Confidence threshold
        
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
            
            # Add class_id label for each box
            class_names = ["FIGHTING", "person_on_floor", "threat_position"]  # Replace with actual class names
            label = f"{class_names[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


# if __name__ == "__main__":
    
#     image_path = r"sample2.jpg"
#     Frame = cv2.imread(image_path)
        
#     frame = YOLO_process_image(Frame, model)

#     # Display the image using matplotlib
#     plt.imshow(frame)
#     plt.show()
