# app.py

import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import logging

app = FastAPI()

# Mount static files to serve HTML page for camera capture
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the MiDaS model from PyTorch Hub
model_type = "DPT_Hybrid"  # Model type for depth estimation
model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
model.eval()
device = torch.device('cpu')  # Use CPU
model.to(device)

# Define the transformation required for the MiDaS input
transform = Compose([
    Resize((384, 384)),  # Resize to 384x384 pixels
    ToTensor(),  # Convert the image to a tensor
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def estimate_depth(frame):
    """Estimate the depth of a frame using the MiDaS model."""
    pil_image = Image.fromarray(frame)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = model(input_tensor)

    # Resize depth map to original frame size
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(frame.shape[0], frame.shape[1]),
        mode='bicubic',
        align_corners=False
    ).squeeze().cpu().numpy()
    return depth_map

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for receiving and processing video frames."""
    await websocket.accept()
    try:
        while True:
            # Receive frame from the client
            data = await websocket.receive_bytes()

            # Decode the frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Estimate depth
            depth_map = estimate_depth(frame)
            
            # Normalize and convert depth map for visualization
            depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_MAGMA)

            # Encode the depth map back to bytes
            _, buffer = cv2.imencode('.jpg', depth_map_colored)
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        logging.info("Client disconnected.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
