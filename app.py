import cv2
import sys


def out(x):
    with open("example.txt", "a") as f:
        f.write(str(x.tolist()))


def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    for i in range(2):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Hand tracking logic goes here
        # ...

        # Display the resulting frame
        out(frame)
        print(frame.shape)
        cv2.imshow('Hand Tracking', frame)

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
