import cv2
import numpy as np
import pyaudio
from yolo import YOLO
import threading
from threading import Event
import time

yolo = YOLO("models/cross-hands-tiny-prn.cfg",
            "models/cross-hands-tiny-prn.weights", ["hand"])
# Define your sine and saw wave generation functions here


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


def generate_sine_wave(stop_event, frequency=440.0, sample_rate=44100, volume=0.5):
    p = pyaudio.PyAudio()
    # range [0.0, 1.0]
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    # Generate samples
    samples = (volume * 32767 * np.sin(2 * np.pi * np.arange(sample_rate)
               * frequency / sample_rate)).astype(np.int16)

    while not stop_event.is_set():
        stream.write(samples.tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()


def generate_saw_wave(stop_event, frequency=440.0, sample_rate=44100, volume=0.2):
    p = pyaudio.PyAudio()
    # range [0.0, 1.0]
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    t = 0  # time counter
    buffer_duration = 0.1  # duration of the buffer (in seconds)
    buffer_size = int(sample_rate * buffer_duration)  # size of the buffer

    while not stop_event.is_set():
        # Generate a buffer of samples
        samples = np.array([2 * (i / sample_rate * frequency - np.floor(
            0.5 + i / sample_rate * frequency)) for i in range(t, t + buffer_size)])
        t += buffer_size  # Increment time counter

        # Scale to 16-bit range and convert to bytes
        int_samples = (volume * 32767 * samples).astype(np.int16).tobytes()

        # Write buffer to PyAudio stream
        stream.write(int_samples)

    stream.stop_stream()
    stream.close()
    p.terminate()


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

while rval:
    frame = cv2.flip(frame, 1)
    width, height, inference_time, results = yolo.inference(frame)

    # display fps
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS',
                (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.rectangle(frame, (0, 0), (340, 720), (0, 255, 255), 2)
    cv2.rectangle(frame, (341, 0), (939, 720), (66, 255, 167), 2)
    cv2.rectangle(frame, (940, 0), (1280, 720), (255, 64, 143), 2)
    # sort by confidence
    results.sort(key=lambda x: x[2])

    # how many hands should be shown
    # hand_count = len(results)
    # if hands != -1:
    #     hand_count = int(hands)

    # display hands
    for detection in results[:hands]:
        id, name, confidence, x, y, w, h = detection
        area = w * h
        cx = x + (w / 2)
        cy = y + (h / 2)
        sine = False
        saw = False
        # draw a bounding box rectangle and label on the image
        if x >= 0 and x <= 340:
            color = (0, 255, 255)
            sine = True
            saw = False
        elif x > 340 and x < 940:
            color = (66, 255, 167)
            sine = False
            saw = False
        elif x >= 940 and x <= 1280:
            color = (255, 64, 143)
            saw = True
            sine = False

        norm_area = area/921600
        if sine:
            if saw_thread and saw_thread.is_alive():
                saw_thread.stop()
                saw_thread.join()
            if not sine_thread or not sine_thread.is_alive():
                sine_thread = StoppableThread(
                    target=generate_sine_wave)
                sine_thread.start()
        elif saw:
            if sine_thread and sine_thread.is_alive():
                sine_thread.stop()
                sine_thread.join()
            if not saw_thread or not saw_thread.is_alive():
                saw_thread = StoppableThread(
                    target=generate_saw_wave)
                saw_thread.start()

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(frame, f'W: {w} H: {h}\nX: {x} Y: {y}',
                    (1000, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("preview", frame)

    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

if sine_thread and sine_thread.is_alive():
    sine_thread.stop()
    sine_thread.join()
if saw_thread and saw_thread.is_alive():
    saw_thread.stop()
    saw_thread.join()

cv2.destroyWindow("preview")
vc.release()
