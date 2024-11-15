from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivymd.app import MDApp
from kivymd.uix.button import MDFillRoundFlatButton, MDIconButton
from kivymd.uix.label import MDLabel
from kivy.graphics.texture import Texture
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from gtts import gTTS
import os
from playsound import playsound


KV = '''

BoxLayout:
    orientation: 'vertical'
    spacing: '20dp'
    padding: '20dp'

    Image:
        id: camera_view
        size_hint_y: None
        height: root.height - dp(248)
        allow_stretch: True
        keep_ratio: True

    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: None
        height: '48dp'
        spacing: '10dp'

        MDFillRoundFlatButton:
            id: detect_button
            text: ''
            on_press: app.toggle_detection()
            size_hint_x: 0.5
            size_hint_y: None
            height: '48dp'
            MDIconButton:
                icon: "play"
                pos: self.parent.x + dp(10), self.parent.center_y - dp(12)
                user_font_size: "24sp"

        MDFillRoundFlatButton:
            id: stop_button
            text: ''
            on_press: app.stop_detection()
            size_hint_x: 0.5
            size_hint_y: None
            height: '48dp'
            md_bg_color: 0, 44, 12, 1
            MDIconButton:
                icon: "microphone"
                pos: self.parent.x + dp(10), self.parent.center_y - dp(12)
                user_font_size: "24sp"

    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: None
        height: '48dp'
        spacing: '10dp'

        MDFillRoundFlatButton:
            id: clear_button
            text: ''
            on_press: app.clear_predicted_text()
            size_hint_x: 0.5
            size_hint_y: None
            height: '48dp'
            md_bg_color: 255, 0, 0, 1
            MDIconButton:
                icon: "delete"
                size_hint_x: None
                size_hint_y: None
                size: dp(24), dp(24)
                pos: self.parent.x + dp(10), self.parent.center_y - dp(12)
                user_font_size: "24sp"

        MDFillRoundFlatButton:
            id: delete_last_button
            text: ''
            on_press: app.delete_last_character()
            size_hint_x: 0.5
            size_hint_y: None
            height: '48dp'
            md_bg_color: 255, 0, 0, 1
            MDIconButton:
                icon: "close-circle"
                pos: self.parent.x + dp(10), self.parent.center_y - dp(12)
                user_font_size: "24sp"

    MDLabel:
        id: landmarks_label
        text: ''
        size_hint_y: None
        height: self.texture_size[1]
        halign: 'center'
        canvas.before:
            Color:
                rgba: 0, 0, 0, 1  # Black color
            Line:
                points: self.x, self.y, self.x + self.width, self.y
                width: 1

    MDLabel:
        id: error_label
        text: ''
        size_hint_y: None
        height: self.texture_size[1]
        halign: 'center'
'''


class HandLandmarksApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Sign Speak"
        self.icon = "logo.png"
        self.camera = cv2.VideoCapture(0)
        self.is_detecting = False
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.labels_dict = {0: 'A', 1: 'B', 2: 'C',3: 'D',4: 'E',5: 'F',6: 'G',7: 'H',8: 'I', 9:'J',
               10:'K' , 11:'L' , 12:'M', 13:'N' ,14:'O' ,15: 'P' ,16:'Q',17:'R' ,18:'S',19:'T',
                20:'U', 21:'V',22:'W',23:'X',24:'Y',25:'Z'}
       
        self.model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = self.model_dict['model']
        self.predicted_characters = ''
        self.last_prediction_time = 0

    def build(self):
        return Builder.load_string(KV)

    def toggle_detection(self):
        detect_button = self.root.ids.detect_button
        if self.is_detecting:
           
           # detect_button.text = 'Démarrer la détection'
            self.is_detecting = False
            self.root.ids.landmarks_label.text = ''
        else:
            
            #detect_button.text = 'Arrêter la détection'
            self.is_detecting = True

    def stop_detection(self):
        self.is_detecting = False
        self.say_characters()

    def clear_predicted_text(self):
        self.predicted_characters = ''
        self.root.ids.landmarks_label.text = ''
        self.root.ids.error_label.text = ''  # Clear any previous errors
    
    def delete_last_character(self):
        if self.predicted_characters:
            self.predicted_characters = self.predicted_characters[:-1]
            self.root.ids.landmarks_label.text = self.predicted_characters
            self.root.ids.error_label.text = ''  # Clear any previous errors
        else:
            self.root.ids.error_label.text = 'Erreur : Aucun caractère à supprimer'

    def say_characters(self):
        if self.predicted_characters:
            text_to_speak = str(self.predicted_characters)
            tts = gTTS(text=self.predicted_characters.lower(), lang='fr')
            tts.save("output.mp3")
            if os.path.exists("output.mp3"):
                playsound("output.mp3")
                os.remove("output.mp3")
                self.predicted_characters = ''
                self.root.ids.error_label.text = ''  
            else:
                self.root.ids.error_label.text = 'Erreur : fichier introuvable'
        else:
            self.root.ids.error_label.text = 'Erreur : Pas de texte à prononcer'

    def update(self, dt):
        ret, frame = self.camera.read()
        if ret and self.is_detecting:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks and time.time() - self.last_prediction_time > 1:  # Delay of 1 second
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                    prediction = self.model.predict([np.asarray(data_aux)])
                    self.predicted_characters += self.labels_dict[int(prediction[0])]
                self.root.ids.landmarks_label.text = self.predicted_characters
                self.last_prediction_time = time.time()

            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Convert the frame to texture and display it in the image widget
        if ret:
            texture = self.texture_from_frame(frame)
            self.root.ids.camera_view.texture = texture

    def texture_from_frame(self, frame):
        # Flip frame vertically to correct camera reversal
        frame = cv2.flip(frame, 0)
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def on_start(self):
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        # Clear the error label on start
        self.root.ids.error_label.text = ''

    def on_stop(self):
        self.camera.release()

if __name__ == '__main__':
    HandLandmarksApp().run()
