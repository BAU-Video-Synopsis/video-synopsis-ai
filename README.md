# 🎥 Video Synopsis of Security Cams

A capstone project focused on creating an AI-powered video synopsis system that condenses long security camera footage into short, informative highlight videos using computer vision and deep learning.

## 🔍 Overview

Security cameras generate hours of video footage daily, making manual review time-consuming and inefficient. Our system solves this by implementing a video synopsis technique that extracts and compiles only the significant events—such as human movement—into a concise summary video.

The project is built using a modular architecture involving:
- **Frontend**: React.js
- **Backend**: Node.js + Express
- **Database**: MongoDB
- **AI Module**: YOLOv8, OpenCV, PyTorch

## 🧠 AI & Computer Vision

The AI module was developed using:
- **YOLOv8** for real-time object detection
- **OpenCV** for image and video processing (frame extraction, motion analysis, etc.)
- **PyTorch** for model training and fine-tuning
- **FFmpeg** for video encoding and optimization

The system identifies and tracks moving objects (primarily humans), and merges key moments into a summarized clip, preserving spatial-temporal integrity.

## 🔗 Key Features

- 🎯 Real-time object detection and tracking
- 📦 Video summarization using frame-based motion analysis
- ⚡ Fast video processing with optimized pipelines
- 🔐 Privacy-focused and scalable
- 💻 User-friendly web interface for uploading and viewing videos

## 🚀 Project Goals

- Automate video review for surveillance footage
- Reduce review time and storage needs
- Enable faster identification of critical moments in long videos
- Build a deployable, user-friendly platform for end users

## 🧪 Tech Stack

| Layer      | Technologies                        |
|------------|-------------------------------------|
| Frontend   | React.js                            |
| Backend    | Node.js, Express                    |
| Database   | MongoDB                             |
| AI Module  | YOLOv8, PyTorch, OpenCV, FFmpeg     |
| Deployment | GitHub, Docker (optional)           |

## 🧑‍💻 Contributors

- Hasan Kemal Mete *(Computer Vision & AI)*
- Bengühan Şahin *(Computer Vision & AI)*
- Ekrem Bulut *(Software Engineering)*
- Doğa Yıldız *(Frontend & Backend Development)*
- Sarper Sarp *(Software Architecture & Integration)*
