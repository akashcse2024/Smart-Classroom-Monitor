import os
import warnings
import time
import datetime
import csv
import json
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from threading import Lock

# --- FLASK SETUP ---
app = Flask(__name__)
app.secret_key = 'smart_classroom_secret_key_2024'  # Change this in production

# --- CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Global Variables
CURRENT_MODE = "LECTURE"
FPS = 30

# SENSITIVITY
EAR_THRESHOLD = 0.19
LOOK_SIDE_LIMIT = 10
LOOK_DOWN_LIMIT = 10

# TIMERS
HEAD_WAIT_FRAMES = 5 * FPS
EYE_WAIT_FRAMES = 5 * FPS
MAX_TURNS_ALLOWED = 5

# SYNCED SCORE DECAY
SLEEP_DECAY = 100 / EYE_WAIT_FRAMES
DISTRACTION_DECAY = 100 / HEAD_WAIT_FRAMES

# Thread-safe data storage
data_lock = Lock()
current_metrics = {
    "studentCount": 0,
    "engagement": 100,
    "mode": "LECTURE",
    "alerts": []
}

# --- LOGGING SETUP ---
session_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"Report_{session_time}.csv"
current_folder = os.getcwd()
full_path = os.path.join(current_folder, log_file)

print(f"\n📂 REPORT WILL BE SAVED AT: {full_path}\n")

# Write Headers
with open(log_file, mode='w', newline='') as f:
    csv.writer(f).writerow(["Timestamp", "Student_ID", "Mode", "Event", "Details"])

def log_event(student_id, mode, event, details):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    with open(log_file, mode='a', newline='') as f:
        csv.writer(f).writerow([now, f"Student_{student_id}", mode, event, details])
    
    # Add to alerts
    with data_lock:
        alert_msg = f"[{now}] Student {student_id}: {event}"
        current_metrics["alerts"].insert(0, alert_msg)
        if len(current_metrics["alerts"]) > 10:
            current_metrics["alerts"] = current_metrics["alerts"][:10]

# --- AI MODELS ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=10,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)
grey_spec = mp_drawing.DrawingSpec(color=(192, 192, 192), thickness=1, circle_radius=1)

# --- CAMERA SETUP ---
camera = None
camera_lock = Lock()
face_data = {}

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        with camera_lock:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                time.sleep(1.0)
                if not camera.isOpened():
                    camera = cv2.VideoCapture(0)
                    time.sleep(1.0)
    return camera

def calculate_ear(eye_points, landmarks, img_w, img_h):
    coords = []
    for id in eye_points:
        lm = landmarks[id]
        coords.append((int(lm.x * img_w), int(lm.y * img_h)))
    A = dist.euclidean(coords[1], coords[5])
    B = dist.euclidean(coords[2], coords[4])
    C = dist.euclidean(coords[0], coords[3])
    return (A + B) / (2.0 * C)

def generate_frames():
    global CURRENT_MODE, face_data
    
    cap = get_camera()
    
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        
        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        dashboard_status = "ACTIVE CLASS"
        dashboard_score = 100
        student_count = 0
        
        if results.multi_face_landmarks:
            student_count = len(results.multi_face_landmarks)
            
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                if i not in face_data:
                    face_data[i] = {
                        'head_timer': 0, 'eye_timer': 0,
                        'score': 100.0,
                        'turn_count': 0, 'is_turning': False,
                        'distraction_logged': False, 'sleep_logged': False,
                        'malpractice_permanent': False, 'malpractice_logged': False
                    }
                
                # Draw face mesh
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, None, grey_spec)
                
                # Calculate head pose
                face_3d, face_2d = [], []
                point_ids = [33, 263, 1, 61, 291, 199]
                for idx in point_ids:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
                
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                x_angle = angles[0] * 360
                y_angle = angles[1] * 360
                
                # Calculate EAR
                left_eye = [362, 385, 387, 263, 373, 380]
                right_eye = [33, 160, 158, 133, 153, 144]
                avg_ear = (calculate_ear(left_eye, face_landmarks.landmark, img_w, img_h) +
                          calculate_ear(right_eye, face_landmarks.landmark, img_w, img_h)) / 2.0
                
                status = "FOCUSED"
                color = (0, 255, 0)
                stats_text = ""
                
                # --- MODE 1: LECTURE ---
                if CURRENT_MODE == "LECTURE":
                    face_data[i]['turn_count'] = 0
                    face_data[i]['malpractice_permanent'] = False
                    
                    # Sleep Check
                    if x_angle > -25:
                        if avg_ear < EAR_THRESHOLD:
                            face_data[i]['eye_timer'] += 1
                            face_data[i]['score'] -= SLEEP_DECAY
                            
                            if face_data[i]['eye_timer'] > EYE_WAIT_FRAMES:
                                status = "SLEEPING!"
                                color = (0, 0, 255)
                                dashboard_status = "ALARM: SLEEPING"
                                if not face_data[i]['sleep_logged']:
                                    log_event(i, "LECTURE", "SLEEPING", f"Timer: {face_data[i]['eye_timer']}")
                                    face_data[i]['sleep_logged'] = True
                            elif face_data[i]['eye_timer'] > 15:
                                status = "Eyes Closed..."
                                color = (0, 255, 255)
                        else:
                            if face_data[i]['eye_timer'] > 0:
                                face_data[i]['eye_timer'] -= 1
                            face_data[i]['sleep_logged'] = False
                    else:
                        face_data[i]['eye_timer'] = 0
                    
                    # Distraction Check
                    if status != "SLEEPING!":
                        if y_angle < -LOOK_SIDE_LIMIT or y_angle > LOOK_SIDE_LIMIT:
                            face_data[i]['head_timer'] += 1
                            face_data[i]['score'] -= DISTRACTION_DECAY
                            
                            if face_data[i]['head_timer'] > HEAD_WAIT_FRAMES:
                                status = "DISTRACTED"
                                color = (0, 165, 255)
                                dashboard_status = "ALARM: TALK"
                                if not face_data[i]['distraction_logged']:
                                    log_event(i, "LECTURE", "DISTRACTED", "Looked away > 5s")
                                    face_data[i]['distraction_logged'] = True
                            else:
                                status = "Looking Away"
                                color = (0, 255, 255)
                        elif x_angle < -LOOK_DOWN_LIMIT:
                            status = "Reading"
                            color = (0, 255, 255)
                        else:
                            if face_data[i]['head_timer'] > 0:
                                face_data[i]['head_timer'] -= 1
                            if face_data[i]['score'] < 100:
                                face_data[i]['score'] += 0.5
                            face_data[i]['distraction_logged'] = False
                    
                    stats_text = f"H: {face_data[i]['head_timer']} | E: {face_data[i]['eye_timer']}"
                
                # --- MODE 2: EXAM ---
                else:
                    if face_data[i]['malpractice_permanent']:
                        status = "MALPRACTICE!"
                        color = (0, 0, 255)
                        dashboard_status = "ALARM: CHEATING"
                    else:
                        if x_angle < -LOOK_DOWN_LIMIT:
                            status = "Writing"
                            color = (0, 255, 255)
                            face_data[i]['is_turning'] = False
                        elif y_angle < -LOOK_SIDE_LIMIT or y_angle > LOOK_SIDE_LIMIT:
                            if not face_data[i]['is_turning']:
                                face_data[i]['turn_count'] += 1
                                face_data[i]['is_turning'] = True
                            
                            if face_data[i]['turn_count'] > MAX_TURNS_ALLOWED:
                                face_data[i]['malpractice_permanent'] = True
                                status = "MALPRACTICE!"
                                color = (0, 0, 255)
                                if not face_data[i]['malpractice_logged']:
                                    log_event(i, "EXAM", "MALPRACTICE", f"Turns: {face_data[i]['turn_count']}")
                                    face_data[i]['malpractice_logged'] = True
                            else:
                                status = f"WARNING ({face_data[i]['turn_count']}/{MAX_TURNS_ALLOWED})"
                                color = (0, 165, 255)
                        else:
                            face_data[i]['is_turning'] = False
                            status = "FOCUSED"
                    
                    stats_text = f"Turns: {face_data[i]['turn_count']}"
                
                # Clamp score
                face_data[i]['score'] = max(0.0, min(100.0, face_data[i]['score']))
                
                # Draw bounding box
                x_vals = [lm.x for lm in face_landmarks.landmark]
                y_vals = [lm.y for lm in face_landmarks.landmark]
                min_x, max_x = int(min(x_vals) * img_w), int(max(x_vals) * img_w)
                min_y, max_y = int(min(y_vals) * img_h), int(max(y_vals) * img_h)
                
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)
                cv2.putText(frame, status, (min_x, min_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, stats_text, (min_x, min_y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                if i == 0:
                    dashboard_score = int(face_data[i]['score'])
        
        # Update metrics
        with data_lock:
            current_metrics["studentCount"] = student_count
            current_metrics["engagement"] = dashboard_score
            current_metrics["mode"] = CURRENT_MODE
        
        # Draw dashboard on frame
        cv2.rectangle(frame, (0, 0), (img_w, 60), (30, 30, 30), -1)
        cv2.putText(frame, f"Students: {student_count}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        (text_w, text_h), _ = cv2.getTextSize(dashboard_status, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        center_x = int((img_w - text_w) / 2)
        status_col = (0, 255, 0) if "ACTIVE" in dashboard_status else (0, 0, 255)
        cv2.putText(frame, dashboard_status, (center_x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_col, 2)
        
        if CURRENT_MODE == "LECTURE":
            sc_col = (0, 255, 0) if dashboard_score > 60 else (0, 0, 255)
            cv2.putText(frame, f"Focus: {dashboard_score}%", (img_w - 180, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sc_col, 2)
        else:
            cv2.putText(frame, "EXAM MODE", (img_w - 180, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ROUTES ---

@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == 'admin' and password == 'teacher':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    if 'logged_in' not in session:
        return "Unauthorized", 401
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics_feed')
def metrics_feed():
    if 'logged_in' not in session:
        return "Unauthorized", 401
    
    def generate():
        while True:
            with data_lock:
                data = json.dumps(current_metrics)
            yield f"data: {data}\n\n"
            time.sleep(0.1)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/set_mode/<mode>', methods=['POST'])
def set_mode(mode):
    global CURRENT_MODE, face_data
    
    if 'logged_in' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    if mode in ["LECTURE", "EXAM"]:
        CURRENT_MODE = mode
        face_data = {}  # Reset face data on mode change
        return jsonify({"success": True, "mode": mode})
    
    return jsonify({"error": "Invalid mode"}), 400

@app.route('/log_attendance', methods=['POST'])
def log_attendance():
    if 'logged_in' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    with data_lock:
        student_count = current_metrics["studentCount"]
    
    log_event("ALL", "ATTENDANCE", "Manual Count", f"Total Students: {student_count}")
    
    with data_lock:
        current_metrics["alerts"].insert(0, f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Attendance logged: {student_count} students")
    
    return jsonify({"success": True, "count": student_count})

if __name__ == '__main__':
    print("=" * 60)
    print("SMART CLASSROOM MONITOR - WEB APPLICATION")
    print("=" * 60)
    print(f"Login credentials: admin / teacher")
    print(f"Starting Flask server...")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
