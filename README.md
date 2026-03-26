# 🎓 Smart Classroom Monitor

An AI-powered, real-time student engagement and behavioral monitoring system built as a Python web application. This system leverages Google's MediaPipe Face Mesh to analyze facial landmarks, head pose orientation, and eye aspect ratios (EAR) to assist educators in tracking classroom engagement and academic integrity.

## 🚀 Features
* **Real-Time AI Tracking:** Simultaneously tracks up to 10 student faces without requiring heavy GPU acceleration.
* **Dual-Mode Logic:**
  * **Lecture Mode:** Detects sleep (via EAR) and distraction (via yaw/pitch angles) to calculate a dynamic 0-100 engagement score.
  * **Exam Mode:** Tracks discrete head-turn frequencies to detect and permanently flag examination malpractice.
* **Live Dashboard:** A premium, dark-themed responsive UI featuring live MJPEG video streaming and zero-latency metric updates via Server-Sent Events (SSE).
* **Automated Logging:** Automatically intercepts critical behavioral alerts and writes them to timestamped CSV reports for post-session analysis.

## 🛠️ Technology Stack
* **Backend:** Python 3.x, Flask
* **Computer Vision:** OpenCV, MediaPipe (Face Mesh), NumPy, SciPy
* **Frontend:** HTML5 (Jinja2), CSS3 (Glassmorphism UI), Vanilla JavaScript
* **Data Flow:** MJPEG (Video), Server-Sent Events / SSE (Metrics)

## ⚙️ How to Run Locally
1. Clone the repository.
2. Create a virtual environment: `python -m venv env1`
3. Activate the environment and install dependencies (OpenCV, MediaPipe, Flask, SciPy, Numpy).
4. Run the application: `python app.py`
5. Open your browser and navigate to `http://localhost:5000`. Default login is `admin` / `teacher`.
