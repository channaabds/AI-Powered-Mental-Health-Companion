import cv2
import numpy as np
import pyaudio
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os
from threading import Thread
import queue
from .utils.feature_extraction import get_audio_features

class MultimodalEmotionDetector:
    def __init__(self):
        # Initialize video-related components
        self.visual_model = load_model("./Codes/Trained_Models/model.keras")
        self.visual_model.load_weights("./Codes/Trained_Models/emotiondetector_weights.weights.h5")
        self.visual_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Initialize audio-related components
        self.audio_model = self._init_audio_model()
        self.audio_model.load_weights("./Codes/Trained_Models/Speech_Emotion_Recognition_Model.h5")
        self.audio_emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")
        
        # Initialize audio recording parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 4
        
        # Initialize display method
        self.use_cv2 = self._check_cv2_display()
        if not self.use_cv2:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Initialize CSV for saving emotions
        self.csv_file = 'multimodal_emotion_data.csv'
        self.last_visual_emotion = None
        self.last_audio_emotion = None
        
        # Create CSV file with headers
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Visual_Emotion', 'Audio_Emotion', 'Combined_Emotion'])
        
        # Audio processing queue
        self.audio_queue = queue.Queue()
        self.is_running = True

    def _init_audio_model(self):
        """Initialize the audio model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(256, 5, padding='same', input_shape=(65, 1)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv1D(128, 5, padding='same'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.MaxPooling1D(pool_size=(8)),
            tf.keras.layers.Conv1D(128, 5, padding='same'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv1D(128, 5, padding='same'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(7),
            tf.keras.layers.Activation('softmax')
        ])
        return model

    def _check_cv2_display(self):
        """Check if OpenCV display works"""
        try:
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow('test', frame)
                cv2.waitKey(1)
                cv2.destroyWindow('test')
                return True
        except cv2.error:
            return False
        return False

    def process_audio(self):
        """Process audio in a separate thread"""
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, 
                       channels=self.CHANNELS,
                       rate=self.RATE,
                       input=True,
                       frames_per_buffer=self.CHUNK)

        while self.is_running:
            frames = []
            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)

            # Process audio data
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_float32 = audio_data.astype(np.float32)
            audio_normalized = audio_float32 / np.iinfo(np.int16).max
            
            # Extract features
            mfccs, _, _, _ = get_audio_features(audio_normalized, self.RATE)
            mfcc_features = np.array(mfccs).reshape(1, 65, 1)

            # Get prediction
            prediction = self.audio_model.predict(mfcc_features, verbose=0)
            emotion_idx = np.argmax(prediction[0])
            emotion = self.audio_emotions[emotion_idx]
            self.audio_queue.put(emotion)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def preprocess_frame(self, frame):
        """Preprocess frame for visual emotion prediction"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)
        return face

    def get_visual_emotion(self, predictions):
        """Convert visual prediction to emotion label"""
        emotion_idx = np.argmax(predictions)
        emotion = self.visual_emotions[emotion_idx]
        confidence = float(predictions[0][emotion_idx])
        return emotion, confidence

    def combine_emotions(self, visual_emotion, audio_emotion):
        """Combine visual and audio emotions using simple rules"""
        if visual_emotion == audio_emotion:
            return visual_emotion
        elif visual_emotion in ['Happy', 'Surprise'] and audio_emotion in ['Happy', 'Surprise']:
            return 'Happy'
        elif visual_emotion in ['Angry', 'Disgust', 'Fear'] and audio_emotion in ['Anger', 'Disgust', 'Fear']:
            return 'Negative'
        elif visual_emotion in ['Sad', 'Neutral'] and audio_emotion in ['Sad', 'Neutral']:
            return 'Sad'
        else:
            return 'Mixed'

    def add_text_to_frame(self, frame, visual_emotion, audio_emotion, combined_emotion):
        """Add emotion text to frame"""
        cv2.putText(frame, f"Visual: {visual_emotion}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Audio: {audio_emotion}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Combined: {combined_emotion}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def save_emotion_data(self, visual_emotion, audio_emotion, combined_emotion):
        """Save emotion data to CSV"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, visual_emotion, audio_emotion, combined_emotion])
        print(f"Saved: Visual={visual_emotion}, Audio={audio_emotion}, Combined={combined_emotion} at {timestamp}")

    def display_frame(self, frame, visual_emotion, audio_emotion, combined_emotion):
        """Display frame with emotions"""
        frame_with_text = self.add_text_to_frame(frame.copy(), visual_emotion, 
                                               audio_emotion, combined_emotion)
        
        if self.use_cv2:
            cv2.imshow("Multimodal Emotion Detector", frame_with_text)
            key = cv2.waitKey(1) & 0xFF
            return key == ord('q')
        else:
            self.ax.clear()
            frame_rgb = cv2.cvtColor(frame_with_text, cv2.COLOR_BGR2RGB)
            self.ax.imshow(frame_rgb)
            self.ax.axis('off')
            plt.pause(0.01)
            return plt.waitforbuttonpress(timeout=0.01)

    def run(self):
        """Main loop for multimodal emotion detection"""
        # Start audio processing thread
        audio_thread = Thread(target=self.process_audio)
        audio_thread.start()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Flip the frame to avoid mirroring
                frame = cv2.flip(frame, 1)

                # Process visual emotion
                face = self.preprocess_frame(frame)
                visual_predictions = self.visual_model.predict(face, verbose=0)
                visual_emotion, _ = self.get_visual_emotion(visual_predictions)

                # Get audio emotion from queue (non-blocking)
                try:
                    audio_emotion = self.audio_queue.get_nowait()
                    self.last_audio_emotion = audio_emotion
                except queue.Empty:
                    audio_emotion = self.last_audio_emotion if self.last_audio_emotion else "Neutral"

                # Combine emotions
                combined_emotion = self.combine_emotions(visual_emotion, audio_emotion)

                # Save to CSV if emotions have changed
                if (visual_emotion != self.last_visual_emotion or 
                    audio_emotion != self.last_audio_emotion):
                    self.save_emotion_data(visual_emotion, audio_emotion, combined_emotion)
                    self.last_visual_emotion = visual_emotion

                # Display results
                should_quit = self.display_frame(frame, visual_emotion, 
                                              audio_emotion, combined_emotion)
                if should_quit:
                    break

        except KeyboardInterrupt:
            print("\nStopping the program...")
        
        finally:
            self.cleanup()
            audio_thread.join()

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        self.cap.release()
        if self.use_cv2:
            cv2.destroyAllWindows()
        else:
            plt.close('all')

if __name__ == "__main__":
    try:
        detector = MultimodalEmotionDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {str(e)}")