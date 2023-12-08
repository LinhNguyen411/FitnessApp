import tkinter as tk
from tkinter import ttk

import threading

import cv2
from PIL import Image, ImageTk

import numpy as np
import tensorflow as tf
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

import time
from datetime import timedelta

class Page:
    def __init__(self, root):
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(pady=5)
    def destroy(self):
        self.main_frame.destroy()
class YogaPage(Page):
    def __init__(self, root):
        super().__init__(root)

        self.pose_time = 60
        self.warmup_time = 20

        self.pose_class_names = ['Warrior One','Warrior Two','Warrior Three','Triangle',
                    'Tree', 'Downward Facing Dog','Upward Facing Dog',
                    'Plank','Bridge',"Child's",'Lotus',
                    'Seated Forward Fold', 'Corpse']

        self.yoga_frame = tk.Frame(self.main_frame)
        self.yoga_frame.pack(padx = 10, pady = 5)

        self.pose_tracker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.model = tf.keras.models.load_model('model.h5')

        self.time_var = tk.StringVar()
        self.command_var = tk.StringVar()
        self.current_pose_var = tk.StringVar()

        self.predict_lock = threading.Lock()
        self.predict_thread = None

        self.update_time_id = None
        self.update_id = None

        self.running = False
        self.elapsed_time = timedelta()

        self.init_value()

        self.camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.img_w = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.img_h = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.draw_content()

    def init_value(self):
        self.predict_label = "Unknown"

        self.current_pose_index = 0

        self.time_var.set("00:00:00")
        self.command_var.set("Welcome")

        self.is_camera_on = False
        self.is_training = False

        self.update()

    def reset_training(self):
        if self.update_id is not None:
            self.main_frame.after_cancel(self.update_id)

        self.btn_start["text"] = "START"

        self.reset_stopwatch()

        self.change_pose_img()

        self.init_value()


    def change_pose_img(self):

        self.current_pose_var.set(self.pose_class_names[self.current_pose_index])

        image = cv2.resize(
            cv2.imread('YogaPose/{}.png'.format(self.current_pose_var.get())),
            (200, 200))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.label.config(image=image)
        self.label.image = image
    def change_pose(self):
        self.current_pose_index += 1
        if self.current_pose_index < len(self.pose_class_names):
            self.change_pose_img()
        else:
            self.reset_training()

    def update_time(self):
        if self.running:
            self.elapsed_time += timedelta(seconds=1)
            self.update_time_id = self.time_label.after(1000, self.update_time)
        else:
            if self.update_time_id is not None:
                self.time_label.after_cancel(self.update_time_id)
                self.update_time_id = None

        self.time_var.set(str(self.elapsed_time)[0:8])
    def start_watch(self):
        self.running = True
        self.update_time()
    def stop_watch(self):
        self.running = False
        self.update_time()
    def reset_stopwatch(self):
        self.stop_watch()
        self.elapsed_time = timedelta()
        self.update_time()

    def draw_content(self):
        top_frame = tk.Frame(self.yoga_frame, highlightbackground='black',
                              highlightthickness=2)
        top_frame.pack()

        left_frame = tk.Frame(top_frame, highlightbackground='black',
                           highlightthickness=2)
        left_frame.pack(side=tk.LEFT, padx = 0, pady = 0)
        left_frame.pack_propagate(False)
        left_frame.configure(width=self.img_w+10, height=self.img_h+10)

        self.cam_screen = ttk.Label(left_frame)
        self.cam_screen.pack(padx=10, pady=10)

        right_frame = tk.Frame(top_frame, highlightbackground='black',
                              highlightthickness=2)
        right_frame.pack(side=tk.LEFT, padx=0, pady=0)
        right_frame.pack_propagate(False)
        right_frame.configure(width=300, height=self.img_h+10)

        self.label = ttk.Label(right_frame)
        self.label.pack(padx = 5, pady = 5)

        self.time_label = tk.Label(right_frame, textvariable=self.current_pose_var, font=("Helvetica", 14))
        self.time_label.pack(pady=10)

        self.change_pose_img()

        self.time_label = tk.Label(right_frame, textvariable=self.time_var, font=("Helvetica", 28))
        self.time_label.pack(pady=10)

        self.command_label = tk.Label(right_frame, textvariable=self.command_var, font=("Helvetica", 28), fg='#00ffff')
        self.command_label.pack(pady=10)

        bottom_frame = tk.Frame(self.yoga_frame, highlightbackground='black',
                               highlightthickness=2)
        bottom_frame.pack(pady = 10)
        bottom_frame.pack_propagate(False)
        bottom_frame.configure(width=self.img_w+300, height=150)

        self.btn_start = tk.Button(bottom_frame, text="START", command=self.toggle_camera, font=("Helvetica", 18), width=10, relief='flat', bg='#00ff00', fg='#000')
        self.btn_start.pack(padx = 5, pady = 5, side = tk.LEFT)

        self.btn_reset = tk.Button(bottom_frame, text="RESET", command=self.reset_training, font=("Helvetica", 18), width=10, relief='flat', bg='#ff0000', fg='#fff')
        self.btn_reset.pack(padx=5, pady=5, side=tk.LEFT)
    def toggle_camera(self):
        self.is_camera_on = not self.is_camera_on
        if self.is_camera_on:
            self.btn_start["text"] = "STOP"
            self.btn_start['bg'] = '#ffff00'
        else:
            self.btn_start["text"] = "START"
            self.btn_start['bg'] = '#00ff00'

    def update(self):
        if self.is_camera_on:
            if self.elapsed_time.total_seconds() == self.pose_time and self.is_training:
                self.reset_stopwatch()
                self.change_pose()
                self.is_training = False
            elif self.elapsed_time.total_seconds() == self.warmup_time and not self.is_training:
                self.reset_stopwatch()
                self.is_training = True

            if self.elapsed_time.total_seconds() < self.warmup_time and not self.is_training and not self.running:
                self.command_var.set("Warm Up")
                self.start_watch()


            _, frame = self.camera.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.is_training:
                image.flags.writeable = False

                result = self.pose_tracker.process(image=image)

                image.flags.writeable = True

                pose_landmarks = result.pose_landmarks

                if pose_landmarks:
                    if not self.running and self.current_pose_var.get() == self.predict_label:
                        self.start_watch()
                    elif self.running and self.current_pose_var.get() != self.predict_label:
                        self.stop_watch()

                    mp_drawing.draw_landmarks(image, pose_landmarks,
                                              mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                     circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )
                    pose_landmarks = [keypoint for landmark in pose_landmarks.landmark for keypoint in
                                          [landmark.x, landmark.y, landmark.z, landmark.visibility]]
                    if not(self.predict_thread and self.predict_thread.is_alive()):
                        self.predict_thread = threading.Thread(target=self.detect, args=(pose_landmarks,))
                        self.predict_thread.start()
                else:
                    self.predict_label = "Unknown"

                if self.predict_label == self.current_pose_var.get():
                    self.command_var.set("Right pose")
                else:
                    self.command_var.set("Wrong pose")

            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            self.cam_screen.config(image=image)
            self.cam_screen.image = image
        else:
            self.stop_watch()

        self.update_id = self.main_frame.after(33, self.update)

    def destroy(self):
        self.camera.release()
        super().destroy()

    def detect(self, kp_list):
        with self.predict_lock:
            kp_list = np.array(kp_list)
            kp_list = np.expand_dims(kp_list, axis=0)
            result = self.model.predict(kp_list)
            self.predict_label = self.pose_class_names[np.argmax(result[0])]

class HomePage(Page):
    def __init__(self,root):
        super().__init__(root)
        self.lb = tk.Label(self.main_frame, text='Home Page\n\nPage: 1', font=('Bold', 30))

        image = cv2.resize(cv2.imread('thumbnail.jpg'),(900,600))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.lb.config(image = image)
        self.lb.image = image
        self.lb.pack(padx = 5, pady = 5)


class FitnessApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.window_width = 1000
        self.window_height = 600
        self.window.geometry('{}x{}'.format(self.window_width, self.window_height))

        self.change_lock = threading.Lock()
        self.change_thread = None

        self.draw_window()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.window.mainloop()
    def draw_window(self):
        self.options_frame = tk.Frame(self.window, bg='#c3c3c3')

        self.home_btn = tk.Button(self.options_frame, text='Home', font=('Bold', 15),
                          fg='#158aff', bd=0, bg='#c3c3c3',
                          command=lambda: self.indicate(self.home_indicate, HomePage))
        self.home_btn.place(x=15, y=50)
        self.home_indicate = tk.Label(self.options_frame, text='', bg='#158aff')
        self.home_indicate.place(x=8, y=50-5, width=5, height=40)

        self.yoga_btn = tk.Button(self.options_frame, text='Yoga', font=('Bold', 15),
                                  fg='#158aff', bd=0, bg='#c3c3c3',
                                  command=lambda: self.indicate(self.yoga_indicate, YogaPage))
        self.yoga_btn.place(x=15, y=100)
        self.yoga_indicate = tk.Label(self.options_frame, text='', bg='#c3c3c3')
        self.yoga_indicate.place(x=8, y=100-5, width=5, height=40)

        self.options_frame.pack(side=tk.LEFT)
        self.options_frame.pack_propagate(False)
        self.options_frame.configure(width=100, height=self.window_height)

        self.main_frame = tk.Frame(root, highlightbackground='black',
                           highlightthickness=2)
        self.main_frame.pack(side=tk.LEFT)
        self.main_frame.pack_propagate(False)
        self.main_frame.configure(width=self.window_width-100, height=self.window_height)

        self.content = HomePage(self.main_frame)
    def hide_indicators(self):
        self.home_indicate.config(bg='#c3c3c3')
        self.yoga_indicate.config(bg='#c3c3c3')
    def indicate(self, lb, page):

        if self.change_thread and self.change_thread.is_alive():
            return
        self.hide_indicators()
        lb.config(bg='#158aff')

        self.change_thread = threading.Thread(target=self.change_page, args=(page,), daemon=True)
        self.change_thread.start()

    def change_page(self, page):
        with self.change_lock:
            self.content.destroy()
            self.content = page(self.main_frame)
    def on_closing(self):
        self.content.destroy()
        self.window.destroy()

root = tk.Tk()
app = FitnessApp(root, "Camera App")