import cv2
import numpy as np
import pyaudio
from yolo import YOLO
import threading
from threading import Event
import pygame
from scipy.signal import butter, lfilter
import time
import os
import wave
yolo = YOLO("models/cross-hands-tiny-prn.cfg",
            "models/cross-hands-tiny-prn.weights", ["hand"])
# Define your sine and saw wave generation functions here

piano_mode = False
filter_mode = False
oscilator_mode = False
locked = False
record = False
audio_initialized = False
stop_audio_playback = False
active_threads = {}
stop_event = threading.Event()
volume_factor = 1.0
cutoff_frequency = 1000


class StoppableThread(threading.Thread):
    def __init__(self, target, args=()):
        super().__init__()
        self._target = target
        self._args = args
        self._stop_event = threading.Event()

    def run(self):
        self._target(*self._args, self._stop_event)

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


def load_audio(file_path):
    # Load the audio file and return the frames as a numpy array
    with wave.open(file_path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        sample_rate = wf.getframerate()
        audio_data = np.frombuffer(frames, dtype=np.int16)
    return audio_data, sample_rate


def start_stream(sample_rate):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True)
    return p, stream


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_lowpass_filter(data, cutoff_freq, fs):
    b, a = butter_lowpass(cutoff_freq, fs)
    y = lfilter(b, a, data)
    return y


def audio_thread(file_path, sample_rate=44100):
    global volume_factor, cutoff_frequency

    audio_data, sample_rate = load_audio(file_path)
    p, stream = start_stream(sample_rate)

    chunk_size = 1024  # or any other suitable size
    for i in range(0, len(audio_data), chunk_size):
        # Apply low-pass filter
        filtered_data = apply_lowpass_filter(
            audio_data[i:i+chunk_size], cutoff_frequency, sample_rate)
        # Adjust the volume
        adjusted_data = (audio_data[i:i+chunk_size]
                         * volume_factor).astype(np.int16)
        stream.write(adjusted_data.tobytes())

        if stop_audio_playback:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()


def generate_sine_wave(degree, sample_rate=44100, volume=0.5):
    frequencies = {1: 220.0,
                   2: 246.94165062806206,
                   3: 277.1826309768721,
                   4: 293.6647679174076,
                   5: 329.6275569128699,
                   6: 369.9944227116344,
                   7: 415.3046975799451,
                   8: 440.0}

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    # Duration for attack and decay in seconds
    attack_duration = 0.3
    decay_duration = 0.3

    # Number of samples for attack and decay
    attack_samples = int(sample_rate * attack_duration)
    decay_samples = int(sample_rate * decay_duration)

    # Create the attack and decay ramps
    attack_ramp = np.linspace(0, 1, attack_samples)
    decay_ramp = np.linspace(1, 0, decay_samples)

    # Generate the sine wave samples
    samples = np.sin(2 * np.pi * np.arange(sample_rate)
                     * frequencies[degree] / sample_rate)

    # Apply the attack ramp
    samples[:attack_samples] *= attack_ramp

    # Apply the decay ramp
    samples[-decay_samples:] *= decay_ramp

    # Scale to the desired volume and convert to int16
    samples = (volume * 32767 * samples).astype(np.int16)

    stream.write(samples.tobytes())  # Play the samples

    stream.stop_stream()
    stream.close()
    p.terminate()


def piano(degree):

    global active_threads

    # Check if there's already a running thread for this degree
    if degree in active_threads and active_threads[degree].is_alive():

        return

    # If not, start a new thread and add it to the dictionary
    thread = threading.Thread(target=generate_sine_wave, args=(degree,))
    thread.start()
    active_threads[degree] = thread

    # Optional: Clean up finished threads
    for d in list(active_threads.keys()):
        if not active_threads[d].is_alive():
            del active_threads[d]


def main():
    global locked, piano_mode, filter_mode, oscilator_mode, piano_state, record, audio_initialized, volume_factor, cutoff_frequency, stop_audio_playback
    piano_thread = None
    # Initialize threading variables
    sine_thread = None
    saw_thread = None

    yolo.size = int(416)
    yolo.confidence = float(0.2)
    hands = 1
    print("starting webcam...")
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()

    else:
        rval = False

    # main loop
    while rval:
        frame = cv2.flip(frame, 1)
        width, height, inference_time, results = yolo.inference(frame)

        # display fps
        cv2.putText(frame, f'{round(1/inference_time,2)} FPS',
                    (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # display piano
        cv2.rectangle(frame, (0, 0), (340, 720), (0, 255, 255), 2)
        cv2.rectangle(frame, (341, 0), (939, 720), (66, 255, 167), 2)
        cv2.rectangle(frame, (940, 0), (1280, 720), (255, 64, 143), 2)

        cv2.rectangle(frame, (940, 0), (1280, 60), (0, 0, 0), -1)

        cv2.rectangle(frame, (940, 60), (1280, 135), (255, 64, 143), 2)
        cv2.rectangle(frame, (940, 135), (1280, 210), (255, 64, 143), 2)
        cv2.rectangle(frame, (940, 210), (1280, 285), (255, 64, 143), 2)
        cv2.rectangle(frame, (940, 285), (1280, 360), (255, 64, 143), 2)
        cv2.rectangle(frame, (940, 360), (1280, 435), (255, 64, 143), 2)
        cv2.rectangle(frame, (940, 435), (1280, 510), (255, 64, 143), 2)
        cv2.rectangle(frame, (940, 510), (1280, 585), (255, 64, 143), 2)
        cv2.rectangle(frame, (940, 585), (1280, 660), (255, 64, 143), 2)

        cv2.rectangle(frame, (940, 660), (1280, 720), (0, 0, 0), -1)

        # display samples
        cv2.rectangle(frame, (0, 60), (340, 135), (255, 255, 200), 2)
        cv2.putText(frame,  'Miguel_Carballo_Barcarola.mp3',
                    (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2)

        # sort by confidence
        results.sort(key=lambda x: x[2])

        # display hands
        if results:
            id, name, confidence, x, y, w, h = results[0]
            area = w * h
            cx = int(x + (w / 2))
            cy = int(y + (h / 2))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 20, (255, 0, 0), 2)
            cv2.putText(frame, f'W: {w} H: {h}\nX: {x} Y: {y}',
                        (1000, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if not locked:
                if piano_thread:
                    # piano_thread.join()
                    stop_event.set()
                    piano_state = "stopped"
                if audio_initialized:
                    stop_audio_playback = True
                    audio_initialized = False

                if cx > 940:
                    piano_mode = True
                    filter_mode = False
                    oscilator_mode = False

                elif cx > 340 and cx < 940:
                    filter_mode = True
                    oscilator_mode = False
                    piano_mode = False

                else:
                    oscilator_mode = True
                    filter_mode = False
                    piano_mode = False

            elif locked and piano_mode:
                if cy > 585 and cy < 660:
                    piano_thread = piano(1)

                elif cy < 585 and cy > 510:
                    piano_thread = piano(2)

                elif cy < 510 and cy > 435:
                    piano_thread = piano(3)

                elif cy < 435 and cy > 360:
                    piano_thread = piano(4)

                elif cy < 360 and cy > 285:
                    piano_thread = piano(5)

                elif cy < 285 and cy > 210:
                    piano_thread = piano(6)

                elif cy < 210 and cy > 135:
                    piano_thread = piano(7)

                elif cy < 135 and cy > 60:
                    piano_thread = piano(8)

                else:
                    if piano_thread:
                        # piano_thread.join()
                        stop_event.set()

                overlay = frame.copy()
                # Draw a rectangle on the overlay
                cv2.rectangle(overlay, (0, 0), (939, 720), (0, 0, 0), -1)
                alpha = 0.5  # For 50% opacity
                # Blend the overlay with the original frame
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Check if recording
                if record:
                    # Position for the circle (e.g., top-right corner)
                    # 50 pixels from the right edge
                    circle_x = frame.shape[1] - 50
                    circle_y = 50  # 50 pixels from the top edge
                    radius = 20  # Circle radius
                    color = (0, 0, 255)  # Red color in BGR
                    thickness = -1  # Negative thickness makes the circle filled

    # Draw a red circle to indicate recording
                    cv2.circle(frame, (circle_x, circle_y),
                               radius, color, thickness)

            elif locked and filter_mode:
                stop_audio_playback = False
                # play "lib/img/audio/Miguel_Carballo_Barcarola.mp3" in a thread
                # the frame has a size of 1280 X 720.
                # consider top right to be highpass filter
                # bottom right is low pass
                # top left is band pass but in the higher frequency range
                # bottom left is the band pass but in the lower freq range
                # map the this to filter the audio file using the variables
                # cx and cy hold the hands center value.
                #
                volume_factor = 1-(cy / 720)
                cutoff_frequency = 500 + (cx / 1280) * 2500
                if not audio_initialized:
                    threading.Thread(target=audio_thread, args=(
                        "lib/img/audio/Miguel_Carballo_Barcarola.wav",)).start()
                    audio_initialized = True

        # Write adjusted data to stream

                overlay = frame.copy()
                # Draw a rectangle on the overlay
                cv2.rectangle(overlay, (0, 0), (340, 720),
                              (255, 255, 128), -1)
                cv2.rectangle(overlay, (939, 0), (1280, 720),
                              (255, 255, 128), -1)
                # Define the opacity factor (between 0 and 1)
                alpha = 0.2  # For 50% opacity
                # Blend the overlay with the original frame
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.imshow("preview", frame)

        rval, frame = vc.read()

        key = cv2.waitKey(20)

        if key == 32:  # Spacebar key code
            locked = not locked  # Toggle the state
        if key == 114:
            record = not record

        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()


if __name__ == "__main__":
    main()
