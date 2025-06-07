
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2 
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, font
from PIL import Image, ImageTk
import threading
import pyttsx3
import queue
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
speech_queue = queue.Queue()

# Load labels for COCO SSD
labels = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "street sign",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella",
    "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk",
    "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "hand", "blender", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier",
]

# Load SSD model
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

class VisionVoxApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionVox 3D")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2a2b2e")  # Dark background
        
        # Variables
        self.running = False
        self.cap = None
        self.log_data = []
        self.last_spoken = {}
        self.detection_enabled = tk.BooleanVar(value=True)
        self.speech_enabled = tk.BooleanVar(value=True)
        
        # Custom 3D wallpaper (abstract tech pattern)
        self.create_3d_wallpaper()
        
        # Custom font
        self.custom_font = font.Font(family="Helvetica", size=12, weight="bold")
        
        # Main container
        self.main_frame = tk.Frame(root, bg="#2a2b2e", bd=0, highlightthickness=0)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Video display (with 3D border)
        self.video_frame = tk.Frame(self.main_frame, bg="#3a3b3e", bd=0, 
                                   highlightbackground="#4a4b4e", highlightthickness=2)
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.video_label = tk.Label(self.video_frame, bg="#3a3b3e")
        self.video_label.pack(padx=10, pady=10)
        
        # Controls frame (with neomorphic buttons)
        self.controls_frame = tk.Frame(self.main_frame, bg="#2a2b2e", bd=0)
        self.controls_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Neomorphic buttons
        self.start_btn = self.create_3d_button(self.controls_frame, "Start", "#4CAF50", self.start_detection)
        self.start_btn.grid(row=0, column=0, padx=10)
        
        self.stop_btn = self.create_3d_button(self.controls_frame, "Stop", "#F44336", self.stop_detection, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=10)
        
        # Checkboxes with modern toggle style
        self.detection_toggle = self.create_toggle_switch(self.controls_frame, "Enable Detection", self.detection_enabled)
        self.detection_toggle.grid(row=0, column=2, padx=10)
        
        self.speech_toggle = self.create_toggle_switch(self.controls_frame, "Enable Speech", self.speech_enabled)
        self.speech_toggle.grid(row=0, column=3, padx=10)
        
        # Log display (3D inset effect)
        self.log_frame = tk.Frame(self.main_frame, bg="#3a3b3e", 
                                highlightbackground="#4a4b4e", highlightthickness=2)
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(self.log_frame, bg="#3a3b3e", fg="white", 
                              insertbackground="white", font=self.custom_font, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(self.log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Save button
        self.save_btn = self.create_3d_button(self.main_frame, "Save Log to CSV", "#2196F3", self.save_log)
        self.save_btn.pack(pady=10)
        
        # Start speech thread
        self.speech_thread = threading.Thread(target=self.speech_worker, daemon=True)
        self.speech_thread.start()
    
    def create_3d_wallpaper(self):
        # Generate a gradient background (fallback if no image)
        try:
            img = Image.open("3d_wallpaper.png")  # Replace with your 3D wallpaper
            img = img.resize((1200, 800), Image.LANCZOS)
            self.bg_image = ImageTk.PhotoImage(img)
            bg_label = tk.Label(self.root, image=self.bg_image)
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except:
            # Fallback gradient
            self.root.configure(bg="#2a2b2e")  # Pure dark if no image
    
    def create_3d_button(self, parent, text, color, command, state=tk.NORMAL):
        # Neomorphic 3D button
        btn = tk.Button(parent, text=text, command=command, state=state,
                       bg=color, fg="white", font=self.custom_font,
                       bd=0, highlightthickness=0,
                       activebackground=color, activeforeground="white",
                       relief=tk.RAISED, padx=20, pady=10)
        
        # Shadow effect
        btn.config(highlightbackground="#1a1b1e", highlightcolor="#1a1b1e")
        return btn
    
    def create_toggle_switch(self, parent, text, var):
        # Modern toggle switch
        frame = tk.Frame(parent, bg="#2a2b2e")
        tk.Label(frame, text=text, bg="#2a2b2e", fg="white", font=self.custom_font).pack(side=tk.LEFT)
        
        toggle = tk.Checkbutton(frame, variable=var, bg="#2a2b2e", activebackground="#2a2b2e",
                              selectcolor="#4CAF50", bd=0, highlightthickness=0)
        toggle.pack(side=tk.LEFT, padx=5)
        return frame
    
    def start_detection(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera!")
                return

            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.update_video()
    
    def stop_detection(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def detect_objects(self, frame):
        if not self.detection_enabled.get():
            return frame, []
            
        img_resized = cv2.resize(frame, (320, 320))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor([img_rgb], dtype=tf.uint8)
        detections = model(input_tensor)
        
        h, w, _ = frame.shape
        boxes = detections['detection_boxes'][0].numpy()
        class_ids = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()
        
        detected_objects = []
        
        for i in range(len(scores)):
            if scores[i] > 0.5:
                box = boxes[i]
                class_id = class_ids[i]
                y1, x1, y2, x2 = box
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

                label = f"{labels[class_id]}: {scores[i]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detected_objects.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Object": labels[class_id],
                    "Confidence": round(float(scores[i]), 2)
                })
                
                # Add to speech queue if not recently spoken
                current_time = time.time()
                if (self.speech_enabled.get() and 
                    labels[class_id] != "background" and 
                    (labels[class_id] not in self.last_spoken or 
                     current_time - self.last_spoken[labels[class_id]] > 5)):
                    speech_queue.put(f"Detected {labels[class_id]}")
                    self.last_spoken[labels[class_id]] = current_time
        
        return frame, detected_objects
    
    def update_video(self):
        if self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame, new_detections = self.detect_objects(frame)
                self.log_data.extend(new_detections)
                
                # Update log display
                for detection in new_detections:
                    self.log_text.config(state=tk.NORMAL)
                    self.log_text.insert(tk.END, f"{detection['Time']} - {detection['Object']} ({detection['Confidence']})\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state=tk.DISABLED)
                
                # Convert frame to PhotoImage
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Keep reference to avoid garbage collection
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            
            self.root.after(10, self.update_video)
    
    def speech_worker(self):
        while True:
            try:
                text = speech_queue.get(timeout=1)
                engine.say(text)
                engine.runAndWait()
            except queue.Empty:
                continue
    
    def save_log(self):
        if self.log_data:
            df = pd.DataFrame(self.log_data)
            df.to_csv("detected_objects.csv", index=False)
            messagebox.showinfo("Success", "Detection log saved to detected_objects.csv")
        else:
            messagebox.showwarning("Warning", "No detection data to save")
    
    def on_closing(self):
        self.stop_detection()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = VisionVoxApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
