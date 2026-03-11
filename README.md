# Theft-Detection

A real-time shoplifting detection system that processes video streams to identify suspicious behavior. The backend is built with FastAPI and runs a fine-tuned YOLOv11 object detection model. The frontend provides a live video dashboard with real-time alerts via WebSockets, including audio notifications and email alerts with full-frame attachments.

## Dataset
This project uses the **Shoplifting** dataset (version 3) from Roboflow, created by `mscprojects`.
- **Project**: shoplifting-cuzf8
- **Classes**: `normal`, `shoplifting`
- **Link**: [Roboflow Universe - Shoplifting Dataset](https://universe.roboflow.com/mscprojects/shoplifting-cuzf8/dataset/3)