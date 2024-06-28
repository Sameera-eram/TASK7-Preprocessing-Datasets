import cv2
import numpy as np
from keras.api.models import model_from_json
import tkinter as tk
from PIL import Image, ImageTk

def load_model(model_json_path, model_weights_path):
    json_file = open(model_json_path, "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(model_weights_path)
    return model



model_a = load_model("model_a.json", "model_weights.weights.h5")
model_b = load_model("combined_model.json", "context_model_weights.weights.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}



root = tk.Tk()
root.title("Real-time Facial Expression Recognition")


def extract_features(frames):
    features = []
    for frame in frames:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        if len(face) == 1:
            (p, q, r, s) = face[0]
            face_img = frame_gray[q:q+s, p:p+r]
            face_img = cv2.resize(face_img, (48, 48))
            features.append(face_img)
        else:
            features.append(np.zeros((48, 48), dtype=np.uint8))
    if len(features) != 5:
        print("Error: Detected less than 5 faces in the sequence.")
        return None
    features = np.array(features)
    features = features.reshape(1, 5, 48, 48, 1)
    features = features / 255.0
    return features


def process_frame():
    _, frame = webcam.read()
    frames.append(frame)
    if len(frames) > 5:
        frames.pop(0)

    features = extract_features(frames)

    if features is not None:
        
        prediction_a = model_a.predict(features)
        prediction_b = model_b.predict(features)

        prediction_label_a = labels[np.argmax(prediction_a)]
        prediction_label_b = labels[np.argmax(prediction_b)]

        label_var.set(f"Predicted emotion (Model A): {prediction_label_a}\nPredicted emotion (Model B): {prediction_label_b}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 480))
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.config(image=imgtk)
    panel.after(10, process_frame)



label_var = tk.StringVar()
label_var.set("Predicted emotion: ")
label = tk.Label(root, textvariable=label_var, font=("Helvetica", 16))
label.pack(pady=20)

panel = tk.Label(root)
panel.pack()

webcam = cv2.VideoCapture(0)

frames = []
process_frame()
root.mainloop()
webcam.release()
cv2.destroyAllWindows()
