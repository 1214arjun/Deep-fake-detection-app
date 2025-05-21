# 🧠 Deepfake Detection Web App

A Python-based web application that detects potential deepfake videos using facial landmark inconsistencies, powered by **MediaPipe**, **OpenCV**, **Pillow**, and **Flask**.

---

## 📽️ Features

- Upload a `.mp4`, `.avi`, or `.mov` video
- Analyzes facial landmarks using **MediaPipe Face Mesh**
- Calculates facial symmetry ratios and motion blurriness
- Flags suspicious patterns possibly indicative of deepfake
- Returns a confidence score and a simple "real vs. fake" result

---

## 🔧 Technologies Used

- [Python 3.x](https://www.python.org/)
- [Flask](https://flask.palletsprojects.com/)
- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [Pillow (PIL)](https://python-pillow.org/)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/deepfake-detection-app.git
cd deepfake-detection-app
