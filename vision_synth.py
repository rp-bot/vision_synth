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

piano_mode = False
filter_mode = False
oscilator_mode = False
locked = False


stop_event = threading.Event()


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


def generate_sine_wave(degree, sample_rate=44100, volume=0.5):
    global piano_mode

    frequencies = {1: 391.99543598174927,
                   2: 440.0,
                   3: 493.8833012561241,
                   4: 523.2511306011972,
                   5: 587.3295358348151,
                   6: 659.2551138257398,
                   7: 739.9888454232688,
                   8: 783.9908719634985}

    p = pyaudio.PyAudio()
    # range [0.0, 1.0]
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    # Generate samples
    samples = (volume * 32767 * np.sin(2 * np.pi * np.arange(sample_rate)
               * frequencies[degree] / sample_rate)).astype(np.int16)

    while not stop_event.is_set() and piano_mode:
        stream.write(samples.tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()


def piano(degree):

    stop_event.clear()
    thread = threading.Thread(target=generate_sine_wave, args=(degree,))
    thread.start()

    return thread


def main():
    global locked, piano_mode, filter_mode, oscilator_mode, piano_state
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
                # Define the opacity factor (between 0 and 1)
                alpha = 0.5  # For 50% opacity
                # Blend the overlay with the original frame
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.imshow("preview", frame)

        rval, frame = vc.read()

        key = cv2.waitKey(20)

        if key == 32:  # Spacebar key code
            locked = not locked  # Toggle the state

        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()


if __name__ == "__main__":
    main()
