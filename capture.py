import os
import click
import cv2
import time


def capture_and_process_frame(cap: cv2.VideoCapture, max_width=512):
    # capture frame from webcam
    ret, frame = cap.read()

    if not ret:
        return None, None

    # get current dimensions
    height, width = frame.shape[:2]

    # calculate new dimensions
    if width > max_width:
        ratio = max_width / width
        new_width = max_width
        new_height = int(height * ratio)
        frame = cv2.resize(frame, (new_width, new_height))

    return frame


# fmt: off
@click.command()
@click.option("--capture", "-c", type=int, default=0, help="the index of the capture device to use")
@click.argument("directory", type=click.Path(exists=False))
# fmt: on
def capture(capture: int, directory: str) -> None:
    # initialize the webcam
    capture = cv2.VideoCapture(capture)

    # check if the webcam is opened correctly
    if not capture.isOpened():
        print("cannot open webcam")
        raise SystemExit

    # create the directory in which to store the frame
    os.makedirs(directory, exist_ok=True)

    print("starting capture in 5 seconds...")

    # wait for the camera to initialize and adjust light levels
    time.sleep(5)

    print("capturing")
    try:
        while True:
            frame = capture_and_process_frame(capture)

            if frame is not None:
                # save the captured frame
                filename = "frame.jpg"
                cv2.imwrite(os.path.join(directory, filename), frame)

                # wait for 1 second
                cv2.waitKey(1000)
            else:
                print("failed to capture frame")
                break

    except KeyboardInterrupt:
        print("closing...")
    except Exception as e:
        repr(e)
    finally:
        cv2.destroyAllWindows()
        capture.release()


if __name__ == "__main__":
    capture()
