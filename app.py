import os
import cv2
import json
import asyncio
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import torch
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import ultralytics
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging to write to a file
logging.basicConfig(
    filename='suspicious_activity.logs',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

VIDEO_SOURCE = '0'

CONSECUTIVE_FRAMES_THRESHOLD = 3
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

def send_email_alert_sync(timestamp_str: str, image_bytes: bytes):
    sender = os.environ.get("SENDER_EMAIL")
    receiver = os.environ.get("RECEIVER_EMAIL")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 465))
    
    if not sender or not receiver or not password:
        logging.warning("Email alert skipped: Missing SENDER_EMAIL, RECEIVER_EMAIL, or GMAIL_APP_PASSWORD in .env")
        return

    try:
        msg = MIMEMultipart('alternative')
        msg["Subject"] = "🚨 SECURITY ALERT: Suspicious Activity Detected"
        msg["From"] = sender
        msg["To"] = receiver

        # Plain text fallback
        text_content = f"Suspicious activity (Class 1) was detected by the AI model at {timestamp_str}.\nPlease check the dashboard immediately."
        
        # HTML design
        html_content = f"""
        <html>
          <body style="margin: 0; padding: 0; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #0a0a0f; color: #ffffff;">
            <div style="max-width: 600px; margin: 40px auto; background-color: #1a1a24; border-radius: 16px; overflow: hidden; border: 1px solid rgba(255,51,102,0.3); box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
              
              <!-- Header -->
              <div style="background: linear-gradient(135deg, #ff3366 0%, #cc0033 100%); padding: 30px; text-align: center;">
                <h1 style="margin: 0; color: #ffffff; font-size: 24px; letter-spacing: 1px;">🚨 SECURITY ALERT</h1>
              </div>
              
              <!-- Body -->
              <div style="padding: 40px 30px;">
                <h2 style="margin-top: 0; color: #ffffff; font-size: 20px;">Suspicious Activity Detected</h2>
                <p style="color: #888899; font-size: 16px; line-height: 1.6;">
                  The AI model has detected high-confidence suspicious behavior (Class 1) on your video feed.
                </p>
                
                <div style="background-color: rgba(255, 255, 255, 0.05); border-left: 4px solid #ff3366; padding: 15px 20px; border-radius: 4px; margin: 25px 0;">
                  <strong style="color: #ffffff; display: block; margin-bottom: 5px;">Time of Detection:</strong>
                  <span style="color: #00e676; font-family: monospace; font-size: 16px;">{timestamp_str}</span>
                </div>
                
                <p style="color: #888899; font-size: 16px; line-height: 1.6; margin-bottom: 30px;">
                  Please review the live records or logs immediately to verify the situation.
                </p>
              </div>
              
              <!-- Footer -->
              <div style="background-color: #12121a; padding: 20px; text-align: center; border-top: 1px solid rgba(255,255,255,0.05);">
                <p style="margin: 0; color: #555566; font-size: 12px;">This is an automated alert from Theft Detector v1.</p>
              </div>
              
            </div>
          </body>
        </html>
        """
        # Attach parts
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Attach image
        if image_bytes:
            image_part = MIMEImage(image_bytes, name=f"detection_{timestamp_str.replace(':', '-')}.jpg")
            image_part.add_header('Content-ID', '<suspicious_frame>')
            msg.attach(image_part)

        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        logging.info(f"Alert email sent successfully to {receiver} via {smtp_server}:{smtp_port}")
    except Exception as e:
        logging.error(f"Failed to send email alert: {e}")

async def async_send_email_alert(image_bytes: bytes):
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Run the blocking SMTP call in a separate thread so the video stream doesn't freeze
    await asyncio.to_thread(send_email_alert_sync, now_str, image_bytes)

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
                            logging.warning("Suspicious Activity Detected!")
                            
                            # Draw boxes on the full original frame for the email
                            email_frame = frame.copy()
                            for box in results[0].boxes:
                                if int(box.cls[0]) == 1:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cv2.rectangle(email_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cv2.putText(email_frame, "Suspicious Activity", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                    
                            _, img_encoded = cv2.imencode('.jpg', email_frame)
                            alert_image_bytes = img_encoded.tobytes()
                            
                            # Fire off the async email task independently
                            asyncio.create_task(async_send_email_alert(alert_image_bytes))
                            await broadcast_alert({
                                "type": "alert", 
                                "message": "Suspicious Activity Detected!", 
                                "level": "critical"
                            })
                else:
                    class_1_consecutive_frames = 0
                    if suspicious_activity_flag:
                        suspicious_activity_flag = False
                        logging.info("Suspicious Activity Cleared.")
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
