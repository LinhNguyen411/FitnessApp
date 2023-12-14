import tkinter as tk
from tkinter import ttk

import threading
from PIL import Image, ImageTk
import cv2

import numpy as np
import tensorflow as tf
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

from datetime import timedelta
import time

YOGA_MODEL = 'models/yogamodel.h5'
GYM_MODEL = 'models/gymmodel.h5'

class Page(tk.Frame):
    def __init__(self, root):
        super().__init__(
            root,
            bg = 'WHITE'
        )
        self.main_frame = self
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx = 5, pady = 5)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

class HomePage(Page):
    def __init__(self, root):
        super().__init__(root)
        self.lb = tk.Label(self.main_frame)

        image = cv2.resize(cv2.imread('thumbnail.jpg'), (900, 600))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.lb.config(image=image)
        self.lb.image = image
        self.lb.pack(padx=5, pady=5)

class YogaPage(Page):
    def __init__(self, root):
        super().__init__(root)
        self.pose_time = 20
        self.warmup_time = 10

        self.pose_class_names = ['Warrior One', 'Warrior Two', 'Warrior Three', 'Triangle',
                                 'Tree', 'Downward Facing Dog', 'Upward Facing Dog',
                                 'Plank', 'Bridge', "Child's", 'Lotus',
                                 'Seated Forward Fold', 'Corpse']

        self.pose_tracker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.model = tf.keras.models.load_model(YOGA_MODEL)

        self.command_var = tk.StringVar()
        self.current_pose_var = tk.StringVar()
        self.track_pose = tk.StringVar()

        self.predict_lock = threading.Lock()
        self.predict_thread = None

        self.update_id = None

        self.update_time_id = None
        self.time_var = tk.StringVar()
        self.running = False
        self.elapsed_time = timedelta()

        self.init_value()

        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.img_w = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.img_h = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.draw_content()

    def init_value(self):
        self.predict_label = "Unknown"

        self.current_pose_index = 0

        self.countdown = 3

        self.time_var.set("00:00:00")
        self.command_var.set("Welcome")

        self.is_camera_on = False
        self.is_training = False

        self.update()

    def reset_training(self):
        if self.update_id is not None:
            self.main_frame.after_cancel(self.update_id)

        self.is_camera_on = True
        self.toggle_camera()

        self.reset_stopwatch()

        self.change_pose_img()

        self.init_value()

    def change_pose_img(self):

        self.current_pose_var.set(self.pose_class_names[self.current_pose_index])

        self.track_pose.set(
            "{}/{} {}".format(self.current_pose_index + 1, len(self.pose_class_names), self.current_pose_var.get()))

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
        top_frame = tk.Frame(self.main_frame, highlightbackground='black',
                             highlightthickness=2)
        top_frame.pack()

        left_frame = tk.Frame(top_frame, highlightbackground='black',
                              highlightthickness=2)
        left_frame.pack(side=tk.LEFT, padx=0, pady=0)
        left_frame.pack_propagate(False)
        left_frame.configure(width=self.img_w + 10, height=self.img_h + 10)

        self.cam_screen = ttk.Label(left_frame)
        self.cam_screen.pack(padx=10, pady=10)

        right_frame = tk.Frame(top_frame, highlightbackground='black',
                               highlightthickness=2)
        right_frame.pack(side=tk.LEFT, padx=0, pady=0)
        right_frame.pack_propagate(False)
        right_frame.configure(width=300, height=self.img_h + 10)

        self.label = ttk.Label(right_frame)
        self.label.pack(padx=5, pady=5)

        self.track_pose_label = tk.Label(right_frame, textvariable=self.track_pose, font=("Helvetica", 12))
        self.track_pose_label.pack(pady=10)

        self.change_pose_img()

        self.time_label = tk.Label(right_frame, textvariable=self.time_var, font=("Helvetica", 28))
        self.time_label.pack(pady=10)

        self.command_label = tk.Label(right_frame, textvariable=self.command_var, font=("Helvetica", 22), fg='#158aff')
        self.command_label.pack(pady=10)

        bottom_frame = tk.Frame(self.main_frame, highlightbackground='black',
                                highlightthickness=2)
        bottom_frame.pack(pady=10)
        bottom_frame.pack_propagate(False)
        bottom_frame.configure(width=self.img_w + 300, height=150)

        self.btn_start = tk.Button(bottom_frame, text="START", command=self.toggle_camera, font=("Helvetica", 18),
                                   width=10, relief='flat', bg='#00ff00', fg='#000')
        self.btn_start.pack(padx=5, pady=5, side=tk.LEFT)

        self.btn_reset = tk.Button(bottom_frame, text="RESET", command=self.reset_training, font=("Helvetica", 18),
                                   width=10, relief='flat', bg='#ff0000', fg='#fff')
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
                self.countdown = 0
                self.is_training = False
            elif self.elapsed_time.total_seconds() == self.warmup_time and not self.is_training:
                self.reset_stopwatch()
                self.is_training = True

            if self.elapsed_time.total_seconds() < self.warmup_time and not self.is_training:
                if not self.running:
                    self.start_watch()
                    self.command_var.set("Warm Up")
                if self.warmup_time - self.elapsed_time.total_seconds() == self.countdown:
                    value = str(self.countdown)
                    self.command_var.set(value)
                    self.countdown -= 1

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
                    if not (self.predict_thread and self.predict_thread.is_alive()):
                        self.predict_thread = threading.Thread(target=self.detect, args=(pose_landmarks,))
                        self.predict_thread.start()
                else:
                    self.predict_label = "Unknown"

                if self.predict_label == self.current_pose_var.get():
                    self.command_var.set("keep posture")
                else:
                    self.command_var.set("Wrong posture")

            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            self.cam_screen.config(image=image)
            self.cam_screen.image = image
        else:
            self.stop_watch()

        self.update_id = self.main_frame.after(33, self.update)

    def destroy(self):
        self.reset_training()
        self.camera.release()
        super().destroy()

    def detect(self, kp_list):
        with self.predict_lock:
            kp_list = np.array(kp_list)
            kp_list = np.expand_dims(kp_list, axis=0)
            result = self.model.predict(kp_list)
            self.predict_label = self.pose_class_names[np.argmax(result[0])]

class GymPage(Page):

    def __init__(self, root):
        super().__init__(root)

        self.rest_time = 10

        self.joints = {
            'ArmRaise': {
                'ANGLE': [20, 120],
                'STAGE': ['DOWN', 'UP'],
                'LEFT': [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                         mp_pose.PoseLandmark.LEFT_HIP.value],
                'RIGHT': [mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                          mp_pose.PoseLandmark.RIGHT_HIP.value]
            },
            'BicycleCrunch': {
                'ANGLE': [60, 160],
                'STAGE': ['LEFT', 'RIGHT'],
                'LEFT': [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value,
                         mp_pose.PoseLandmark.LEFT_ANKLE.value],
                'RIGHT': [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value,
                          mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            },
            'BirdDog': {
                'ANGLE': [90, 160],
                'STAGE': ['DOWN', 'UP'],
                'LEFT': [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                         mp_pose.PoseLandmark.LEFT_HIP.value],
                'RIGHT': [mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                          mp_pose.PoseLandmark.RIGHT_HIP.value]
            },
            'Curl': {
                'ANGLE': [170, 30],
                'STAGE': ['LEFT', 'RIGHT'],
                'LEFT': [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value,
                         mp_pose.PoseLandmark.LEFT_WRIST.value],
                'RIGHT': [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                          mp_pose.PoseLandmark.RIGHT_WRIST.value]
            },
            'Fly': {
                'ANGLE': [70, 25],
                'STAGE': ['RETURN', 'PUSH'],
                'LEFT': [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                         mp_pose.PoseLandmark.LEFT_HIP.value],
                'RIGHT': [mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                          mp_pose.PoseLandmark.RIGHT_HIP.value]
            },
            'LegRaise': {
                'ANGLE': [160, 90],
                'STAGE': ['DOWN', 'UP'],
                'LEFT': [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value,
                         mp_pose.PoseLandmark.LEFT_KNEE.value],
                'RIGHT': [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
                          mp_pose.PoseLandmark.RIGHT_KNEE.value]
            },
            'OverHeadPress': {
                'ANGLE': [70, 170],
                'STAGE': ['DOWN', 'UP'],
                'LEFT': [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value,
                         mp_pose.PoseLandmark.LEFT_WRIST.value],
                'RIGHT': [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                          mp_pose.PoseLandmark.RIGHT_WRIST.value]
            },
            'PushUp': {
                'ANGLE': [160, 70],
                'STAGE': ['UP', 'DOWN'],
                'LEFT': [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value,
                         mp_pose.PoseLandmark.LEFT_WRIST.value],
                'RIGHT': [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                          mp_pose.PoseLandmark.RIGHT_WRIST.value]
            },
            'Squat': {
                'ANGLE': [170, 85],
                'STAGE': ['UP', 'DOWN'],
                'LEFT': [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value,
                         mp_pose.PoseLandmark.LEFT_ANKLE.value],
                'RIGHT': [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value,
                          mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            }
        }


        self.pose_class_names = ["ArmRaise", "BicycleCrunch", "BirdDog","Curl", "Fly", "LegRaise", "OverHeadPress", "PushUp", "Squat", "Superman"]

        self.pose_tracker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.model = tf.keras.models.load_model(GYM_MODEL)

        self.command_var = tk.StringVar()
        self.current_pose_name = tk.StringVar()
        self.no_of_sets = tk.StringVar()
        self.no_of_reps = tk.StringVar()

        self.predict_thread = None

        self.animate_thread = None
        self.animate_lock = threading.Lock()

        self.current_pose_name.set(self.pose_class_names[0])

        self.update_time_id = None
        self.update_id = None

        self.predict_label = "Unknown"

        self.is_camera_on = False
        self.is_animate = True
        self.is_training = False

        self.n_time_steps = 10
        self.kp_list = []

        self.update_time_id = None
        self.time_var = tk.StringVar()
        self.running = False
        self.elapsed_time = timedelta()
        self.time_var.set("00:00:00")

        self.time_track = None
        self.sets_track = 1
        self.reps_track = 0
        self.stage_track = None

        self.command_var.set("Welcome")

        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.img_w = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.img_h = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.update()

        self.draw_content()

    def draw_content(self):
        top_frame = tk.Frame(self.main_frame, highlightbackground='black',
                             highlightthickness=2)
        top_frame.pack()

        left_frame = tk.Frame(top_frame, highlightbackground='black',
                              highlightthickness=2)
        left_frame.pack(side=tk.LEFT, padx=0, pady=0)
        left_frame.pack_propagate(False)
        left_frame.configure(width=self.img_w + 10, height=self.img_h + 10)

        self.cam_screen = ttk.Label(left_frame)
        self.cam_screen.pack(padx=10, pady=10)

        self.right_frame = tk.Frame(top_frame, highlightbackground='black',
                                    highlightthickness=2)
        self.right_frame.pack(side=tk.LEFT, padx=0, pady=0)
        self.right_frame.pack_propagate(False)
        self.right_frame.configure(width=300, height=self.img_h + 10)

        self.label = ttk.Label(self.right_frame)
        self.label.pack(padx=5, pady=5)

        self.track_pose_label = tk.Label(self.right_frame, textvariable=self.current_pose_name, font=("Helvetica", 12))
        self.track_pose_label.pack(pady=10)

        self.time_label = tk.Label(self.right_frame, textvariable=self.time_var, font=("Helvetica", 14), bg="#158aff",
                                   fg="#fff")
        self.time_label.pack(pady=10)

        self.command_label = tk.Label(self.right_frame, textvariable=self.command_var, font=("Helvetica", 22),
                                      fg='#158aff')
        self.command_label.pack(pady=10)

        bottom_frame = tk.Frame(self.main_frame, highlightbackground='black',
                                highlightthickness=2)
        bottom_frame.pack(pady=10)
        bottom_frame.pack_propagate(False)
        bottom_frame.configure(width=self.img_w + 300, height=150)

        self.btn_start = tk.Button(bottom_frame, text="START", command=self.toggle_camera, font=("Helvetica", 18),
                                   width=10, relief='flat', bg='#00ff00', fg='#000')
        self.btn_start.pack(padx=5, pady=5, side=tk.LEFT)

        lb = tk.Label(bottom_frame)
        lb.pack(padx=10, side=tk.LEFT)

        combo_label = tk.Label(bottom_frame, text="Exercise:", font=("Helvetica", 14), fg='#158aff')
        combo_label.pack(padx=10, side=tk.LEFT)

        self.my_combo = ttk.Combobox(bottom_frame, values=self.pose_class_names[:-1], font=("Helvetica", 14), width=10)
        self.my_combo.current(0)
        self.my_combo.pack(side=tk.LEFT)
        self.my_combo.bind("<<ComboboxSelected>>", self.change_exercise)

        self.change_label()

        lb = tk.Label(bottom_frame)
        lb.pack(padx=10, side=tk.LEFT)

        sets_label = tk.Label(bottom_frame, text="Sets:", font=("Helvetica", 14), fg='#158aff')
        sets_label.pack(padx=10, side=tk.LEFT)

        self.sets_input = ttk.Entry(bottom_frame, textvariable=self.no_of_sets, width=5, font=("Helvetica", 14))
        self.sets_input.pack(side=tk.LEFT)

        lb = tk.Label(bottom_frame)
        lb.pack(padx=10, side=tk.LEFT)

        reps_label = tk.Label(bottom_frame, text="Reps:", font=("Helvetica", 14), fg='#158aff')
        reps_label.pack(padx=10, side=tk.LEFT)

        self.reps_input = ttk.Entry(bottom_frame, textvariable=self.no_of_reps, width=5, font=("Helvetica", 14))
        self.reps_input.pack(side=tk.LEFT)

    def reset_training(self):
        if self.update_id is not None:
            self.main_frame.after_cancel(self.update_id)

        self.btn_start["text"] = "START"
        self.btn_start['bg'] = '#00ff00'

        self.sets_input.config(state="enabled")
        self.reps_input.config(state="enabled")

        self.command_var.set("Welcome")

        self.is_camera_on = False
        self.is_training = False

        self.reps_track = 0
        self.sets_track = 1

        self.reset_stopwatch()

        self.kp_list = []

        self.update()

    def start_training(self):
        if self.no_of_sets.get() and self.no_of_reps.get() and int(self.no_of_sets.get()) >= 0 and int(
                self.no_of_reps.get()) >= 0:
            self.btn_start["text"] = "STOP"
            self.btn_start['bg'] = '#ff0000'

            self.sets_input.config(state="disabled")
            self.reps_input.config(state="disabled")

            self.is_camera_on = True
            self.is_training = True
            if not self.running:
                self.start_watch()

    def change_exercise(self, e):
        if self.current_pose_name.get() == self.my_combo.get():
            return
        self.change_label()
        self.reset_training()

    def change_label(self):
        if self.animate_thread and self.animate_thread.is_alive():
            self.is_animate = False

        self.current_pose_name.set(self.my_combo.get())
        self.stage_track = self.joints[self.current_pose_name.get()]['STAGE'][0]
        self.animate_thread = threading.Thread(target=self.change_pose_img, daemon=True)
        self.animate_thread.start()

    def change_pose_img(self):
        file = 'GymPose/{}.mp4'.format(self.current_pose_name.get())
        cap = cv2.VideoCapture(file)
        time.sleep(1)
        if not self.is_animate:
            self.is_animate = not self.is_animate
        frame_counter = 0
        while (self.is_animate):
            ret, frame = cap.read()
            if ret:
                frame_counter += 1

                image = cv2.resize(frame, (200, 200))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                time.sleep(0.1)
                self.label.config(image=image)
                self.label.image = image

                if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    frame_counter = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                break

        cap.release()

    def toggle_camera(self):
        if self.is_camera_on:
            self.reset_training()
        else:
            self.start_training()

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

    def calculate_angle(self, landmarks):
        current_pose = self.current_pose_name.get()
        side = 'LEFT'
        if not all([landmarks[joint].visibility > 0.5 for joint in self.joints[current_pose][side]]):
            side = 'RIGHT'

        a = [landmarks[self.joints[current_pose][side][0]].x,
             landmarks[self.joints[current_pose][side][0]].y]
        b = [landmarks[self.joints[current_pose][side][1]].x,
             landmarks[self.joints[current_pose][side][1]].y]
        c = [landmarks[self.joints[current_pose][side][2]].x,
             landmarks[self.joints[current_pose][side][2]].y]

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def update(self):
        if self.is_camera_on:
            _, frame = self.camera.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.is_training:
                image.flags.writeable = False

                result = self.pose_tracker.process(image=image)

                image.flags.writeable = True

                pose_landmarks = result.pose_landmarks

                if pose_landmarks:
                    mp_drawing.draw_landmarks(image, pose_landmarks,
                                              mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                     circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )

                    landmarks = [keypoint for landmark in pose_landmarks.landmark for keypoint in
                                 [landmark.x, landmark.y, landmark.z, landmark.visibility]]
                    self.kp_list.append(landmarks)
                    if len(self.kp_list) == self.n_time_steps:
                        self.predict_thread = threading.Thread(target=self.detect, args=(self.kp_list,))
                        self.predict_thread.start()
                        self.kp_list = []

                    current_pose = self.current_pose_name.get()
                    if self.predict_label == current_pose:
                        angle = self.calculate_angle(pose_landmarks.landmark)
                        first_stage = self.joints[current_pose]['STAGE'][0]
                        second_stage = self.joints[current_pose]['STAGE'][1]
                        first_angle = self.joints[current_pose]['ANGLE'][0]
                        second_angle = self.joints[current_pose]['ANGLE'][1]
                        if first_angle > second_angle:
                            if angle > first_angle and self.stage_track == second_stage:
                                self.stage_track = first_stage
                                self.reps_track += 1
                            if angle < second_angle and self.stage_track == first_stage:
                                self.stage_track = second_stage
                        else:
                            if angle < first_angle and self.stage_track == second_stage:
                                self.stage_track = first_stage
                                self.reps_track += 1
                            if angle > second_angle and self.stage_track == first_stage:
                                self.stage_track = second_stage
                else:
                    self.predict_label = "Unknown"

                if self.predict_label == self.current_pose_name.get():
                    text = 'Set {} of {}\n{} {}'.format(self.sets_track, self.no_of_sets.get(), self.reps_track,
                                                        self.stage_track)
                    self.command_var.set(text)
                else:
                    self.command_var.set("Wrong posture")

                if self.reps_track == int(self.no_of_reps.get()):
                    self.is_training = False

            else:
                if self.sets_track == int(self.no_of_sets.get()):
                    self.stop_watch()
                    self.command_var.set('Well done!')
                    self.btn_start["text"] = "RESET"
                    self.btn_start['bg'] = '#00ffff'
                else:
                    if not self.time_track:
                        self.time_track = self.elapsed_time.total_seconds()

                    self.command_var.set(
                        "Rest\n{}".format(int(self.time_track + self.rest_time - self.elapsed_time.total_seconds())))
                    if self.elapsed_time.total_seconds() - self.time_track == self.rest_time:
                        self.is_training = True
                        self.reps_track = 0
                        self.sets_track += 1
                        self.time_track = None

            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            self.cam_screen.config(image=image)
            self.cam_screen.image = image
        self.update_id = self.main_frame.after(33, self.update)

    def destroy(self):
        self.reset_training()
        self.camera.release()
        self.is_animate = False
        if self.running:
            self.stop_watch()
        if self.update_id:
            self.main_frame.after_cancel(self.update_id)
        super().destroy()

    def detect(self, kp_list):
        kp_list = np.array(kp_list)
        kp_list = np.expand_dims(kp_list, axis=0)
        result = self.model.predict(kp_list)
        self.predict_label = self.pose_class_names[np.argmax(result[0])]

class FitnessApp(tk.Frame):
    def __init__(self,root):
        super().__init__(
            root,
            bg='WHITE'
        )

        self.main_frame = self
        self.main_frame.pack(fill = tk.BOTH, expand = True)
        self.main_frame.columnconfigure(0, weight = 1)
        self.main_frame.rowconfigure(0, weight = 1)
        self.window_width = 1000
        self.window_height = 600

        self.change_thread = None


    def create_initial_widgets(self):
        self.options_frame = tk.Frame(self.main_frame, bg='#158aff')

        self.home_btn = tk.Button(self.options_frame, text='Home', font=('Bold', 15),
                                  fg='#fff', bd=0, bg='#158aff',
                                  command=lambda: self.indicate(self.home_indicate, 'HomePage'))
        self.home_btn.place(x=15, y=50)
        self.home_indicate = tk.Label(self.options_frame, text='', bg='#fff')
        self.home_indicate.place(x=8, y=50 - 5, width=5, height=40)

        self.yoga_btn = tk.Button(self.options_frame, text='Yoga', font=('Bold', 15),
                                  fg='#fff', bd=0, bg='#158aff',
                                  command=lambda: self.indicate(self.yoga_indicate, 'YogaPage'))
        self.yoga_btn.place(x=15, y=100)
        self.yoga_indicate = tk.Label(self.options_frame, text='', bg='#158aff')
        self.yoga_indicate.place(x=8, y=100 - 5, width=5, height=40)

        self.gym_btn = tk.Button(self.options_frame, text='Gym', font=('Bold', 15),
                                 fg='#fff', bd=0, bg='#158aff',
                                 command=lambda: self.indicate(self.gym_indicate, 'GymPage'))
        self.gym_btn.place(x=15, y=150)
        self.gym_indicate = tk.Label(self.options_frame, text='', bg='#158aff')
        self.gym_indicate.place(x=8, y=150 - 5, width=5, height=40)

        self.options_frame.pack(side=tk.LEFT)
        self.options_frame.pack_propagate(False)
        self.options_frame.configure(width=100, height=self.window_height)

        self.content_frame = tk.Frame(self.main_frame, highlightbackground='black',
                                   highlightthickness=2, bg='#fff')
        self.content_frame.pack(side=tk.LEFT)
        self.content_frame.pack_propagate(False)
        self.content_frame.configure(width=self.window_width - 100, height=self.window_height)

        page = HomePage(self.content_frame)
    def hide_indicators(self):
        self.home_indicate.config(bg='#158aff')
        self.yoga_indicate.config(bg='#158aff')
        self.gym_indicate.config(bg='#158aff')

    def indicate(self, lb, page_name):

        if self.change_thread and self.change_thread.is_alive():
            return
        self.hide_indicators()
        lb.config(bg='#fff')
        self.change_thread = threading.Thread(target=self.change_page, args=(page_name,))
        self.change_thread.start()

    def change_page(self, page_name):
        for child in self.content_frame.winfo_children():
            child.destroy()

        if page_name == 'HomePage':
            page = HomePage(self.content_frame)
        elif page_name == 'YogaPage':
            page = YogaPage(self.content_frame)
        elif page_name == 'GymPage':
            page = GymPage(self.content_frame)


if __name__ == '__main__':
    root = tk.Tk()
    root.title('FitnessApp')
    root.geometry('1000x600')
    root.iconbitmap('icon.ico')
    root.resizable(width=False, height=False)
    my_app_instance = FitnessApp(root)
    my_app_instance.create_initial_widgets()
    root.mainloop()
