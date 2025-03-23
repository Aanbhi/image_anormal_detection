from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn
import os
import cv2
import numpy as np
from PIL import Image
import io
import time
from utils import save_uploaded_file, process_image

app = FastAPI(title="Computer Vision Models API")

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

# Configure templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/processed", StaticFiles(directory="processed"), name="processed")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = await save_uploaded_file(file)
        
        # Return the template with the uploaded image path
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "image_path": file_path}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": f"Upload error: {str(e)}"}
        )

@app.post("/process")
async def process_file(request: Request, file_path: str = Form(...), model_name: str = Form(...)):
    try:
        # Process the image with the selected model
        processed_image_path, result_text = await process_image(file_path, model_name)
        
        # Ensure the image path is correctly formatted for the browser
        # The browser needs a path that's relative to the root or an absolute URL
        image_url = f"/{processed_image_path}"
        
        # Return the image path and result text
        return {"image_url": image_url, "result_text": result_text}
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

@app.get("/view-image/{image_path:path}")
async def view_image(image_path: str):
    """Streaming endpoint for viewing full-screen images"""
    try:
        # Read the image
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            
            # Convert image to bytes for streaming
            success, encoded_image = cv2.imencode('.jpg', image)
            if not success:
                raise ValueError("Failed to encode the image")
            
            # Return the image as a streaming response
            return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")
        else:
            return JSONResponse(status_code=404, content={"error": "Image not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/models")
async def get_available_models():
    return {
        "models": [
            {"name": "florence2", "display_name": "Florence-2 Caption"},
            {"name": "owlv2", "display_name": "OWLv2 Object Detection"},
            {"name": "yolov11", "display_name": "YOLOv11 Fighting Detection"}
        ]
    }

# Configure templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/processed", StaticFiles(directory="processed"), name="processed")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)