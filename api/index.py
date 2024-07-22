# # /api/index.py

# from flask import Flask, jsonify

# app = Flask(__name__)


# @app.route("/")
# def home():
#     return "Flask Vercel Example - Hello World", 200


# @app.errorhandler(404)
# def page_not_found(e):
#     return jsonify({"status": 404, "message": "Not Found"}), 404

import os
import cv2
import dlib
import math
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def compare_eye_openness(shape):
    left_eye_dist1 = math.sqrt((shape.part(37).x - shape.part(41).x)**2 + (shape.part(37).y - shape.part(41).y)**2)
    left_eye_dist2 = math.sqrt((shape.part(38).x - shape.part(40).x)**2 + (shape.part(38).y - shape.part(40).y)**2)
    left_eye_avg_distance = (left_eye_dist1 + left_eye_dist2) / 2

    right_eye_dist1 = math.sqrt((shape.part(43).x - shape.part(47).x)**2 + (shape.part(43).y - shape.part(47).y)**2)
    right_eye_dist2 = math.sqrt((shape.part(44).x - shape.part(46).x)**2 + (shape.part(44).y - shape.part(46).y)**2)
    right_eye_avg_distance = (right_eye_dist1 + right_eye_dist2) / 2

    if left_eye_avg_distance > right_eye_avg_distance:
        ratio = (left_eye_avg_distance / right_eye_avg_distance - 1) * 100
        return f"Left eye is {ratio:.2f}% more open than right eye."
    else:
        ratio = (right_eye_avg_distance / left_eye_avg_distance - 1) * 100
        return f"Right eye is {ratio:.2f}% more open than left eye."

def check_face_yaw(shape, left_eye_center_x, left_eye_center_y, right_eye_center_x, right_eye_center_y):
    left_distance = math.sqrt((shape.part(27).x - left_eye_center_x)**2 + (shape.part(27).y - left_eye_center_y)**2)
    right_distance = math.sqrt((shape.part(27).x - right_eye_center_x)**2 + (shape.part(27).y - right_eye_center_y)**2)

    if left_distance > right_distance:
        ratio = (left_distance / right_distance - 1) * 100
        return f"Face Yaw: Left eye to nose distance is {ratio:.2f}% greater than right eye to nose distance"
    elif right_distance > left_distance:
        ratio = (right_distance / left_distance - 1) * 100
        return f"Face Yaw: Right eye to nose distance is {ratio:.2f}% greater than left eye to nose distance"
    else:
        return "Distances are equal"

def calculate_face_roll(eye_centers):
    left_eye_center = eye_centers[0]
    right_eye_center = eye_centers[1]
    angle_rad = math.atan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0])
    angle_deg = math.degrees(angle_rad)
    return f"Face Roll: Angle: {angle_deg:.2f} degrees"

def check_face_pitch(shape, left_eye_center_x, left_eye_center_y, right_eye_center_x, right_eye_center_y):
    x27 = shape.part(27).x
    y27 = shape.part(27).y
    x28 = shape.part(28).x
    y28 = shape.part(28).y

    y_line_at_x27 = left_eye_center_y + (right_eye_center_y - left_eye_center_y) * (x27 - left_eye_center_x) / (right_eye_center_x - left_eye_center_x)
    y_line_at_x28 = left_eye_center_y + (right_eye_center_y - left_eye_center_y) * (x28 - left_eye_center_x) / (right_eye_center_x - left_eye_center_x)

    if y27 > y_line_at_x27:
        return "Facedown"
    elif y28 < y_line_at_x28:
        return "Faceup"
    else:
        return "Normal Pitch"

def check_mouth_openness(shape):
    distance_62_66 = math.sqrt((shape.part(62).x - shape.part(66).x)**2 + (shape.part(62).y - shape.part(66).y)**2)
    distance_51_62 = math.sqrt((shape.part(51).x - shape.part(62).x)**2 + (shape.part(51).y - shape.part(62).y)**2)

    if distance_62_66 > 0.5 * distance_51_62:
        return "Mouth is open."
    else:
        return "Mouth is closed."

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Load the image
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)
        results = []

        for face in faces:
            shape = predictor(gray, face)

            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            left_eye_center_x = (shape.part(37).x + shape.part(38).x + shape.part(40).x + shape.part(41).x) // 4
            left_eye_center_y = (shape.part(37).y + shape.part(38).y + shape.part(40).y + shape.part(41).y) // 4
            right_eye_center_x = (shape.part(43).x + shape.part(44).x + shape.part(46).x + shape.part(47).x) // 4
            right_eye_center_y = (shape.part(43).y + shape.part(44).y + shape.part(46).y + shape.part(47).y) // 4
            eye_centers = [(left_eye_center_x, left_eye_center_y), (right_eye_center_x, right_eye_center_y)]

            cv2.circle(image, (left_eye_center_x, left_eye_center_y), 2, (0, 0, 255), -1)
            cv2.circle(image, (right_eye_center_x, right_eye_center_y), 2, (0, 0, 255), -1)
            cv2.line(image, (left_eye_center_x, left_eye_center_y), (right_eye_center_x, right_eye_center_y), (255, 0, 0), 1)

            results.append(calculate_face_roll(eye_centers))
            results.append(check_face_yaw(shape, left_eye_center_x, left_eye_center_y, right_eye_center_x, right_eye_center_y))
            results.append(check_face_pitch(shape, left_eye_center_x, left_eye_center_y, right_eye_center_x, right_eye_center_y))
            results.append(compare_eye_openness(shape))
            results.append(check_mouth_openness(shape))
        
        result_text = '\n'.join(results)
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
        cv2.imwrite(result_image_path, image)
        print("upload successful")
        return jsonify({'filename': 'result_' + filename, 'result_text': result_text})

@app.route('/uploads/<filename>')
def send_image_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')
    else:
        print("in else")
        return abort(404, description="Resource not found")
        
if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
