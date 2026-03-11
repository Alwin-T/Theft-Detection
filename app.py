import os
import cv2
import json
import asyncio
import torch
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import ultralytics

# PyTorch 2.6+ security update workaround for loading YOLO weights
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

app = FastAPI(title="Suspicious Activity Detector")

templates = Jinja2Templates(directory="templates")

model_path = 'best (2).pt'
model = None
try:
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        print(f"Warning: Model {model_path} not found. Ensure it exists.")
except Exception as e:
    print(f"Error loading model: {e}")

VIDEO_SOURCE = 'dataset/SD/SL/videmmmmmmsss_90.mp4'

CONSECUTIVE_FRAMES_THRESHOLD = 5
class_1_consecutive_frames = 0
suspicious_activity_flag = False

active_connections = []

async def connect(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

def disconnect(websocket: WebSocket):
    if websocket in active_connections:
        active_connections.remove(websocket)

async def broadcast_alert(message: dict):
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            pass

async def async_process_video():
    global class_1_consecutive_frames, suspicious_activity_flag, model
    
    # Reset state on new stream setup
    class_1_consecutive_frames = 0
    suspicious_activity_flag = False
    
    source = int(VIDEO_SOURCE) if str(VIDEO_SOURCE).isdigit() else VIDEO_SOURCE
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error opening video stream or file: {source}")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                if str(source) != '0':
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
                    
            # Run YOLO inference
            if model:
                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()
                
                has_class_1 = False
                for box in results[0].boxes:
                    if int(box.cls[0]) == 1:
                        has_class_1 = True
                        break
                
                if has_class_1:
                    class_1_consecutive_frames += 1
                    if class_1_consecutive_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
                        if not suspicious_activity_flag:
                            suspicious_activity_flag = True
                            await broadcast_alert({
                                "type": "alert", 
                                "message": "Suspicious Activity Detected!", 
                                "level": "critical"
                            })
                else:
                    class_1_consecutive_frames = 0
                    if suspicious_activity_flag:
                        suspicious_activity_flag = False
                        await broadcast_alert({
                            "type": "info", 
                            "message": "Activity cleared.", 
                            "level": "info"
                        })
            else:
                annotated_frame = frame
                cv2.putText(annotated_frame, "Model not loaded", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
            await asyncio.sleep(0.01)
    finally:
        cap.release()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4') or f.endswith('.avi')]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "video_files": video_files,
        "current_source": VIDEO_SOURCE
    })

@app.post("/set_source")
async def set_source(source: str = Form(...)):
    global VIDEO_SOURCE
    VIDEO_SOURCE = source
    # Tell clients source changed
    await broadcast_alert({"type": "info", "message": f"Source changed to {source}", "level": "info"})
    return RedirectResponse(url="/", status_code=303)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(async_process_video(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        disconnect(websocket)
