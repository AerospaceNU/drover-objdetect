import argparse
import cv2 as cv
from gui import Display

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s",
        required=False,
        default=None,
        help="Source path reference for mp4. If none provided, defaults to live camera feed",
    )

    parser.add_argument(
        "--w",
        required=False,
        default=None,
        help="Weight reference for Detection module",
    )

    args = parser.parse_args()

    display = Display(args.s, windowName="Main")

    try:
        while True:
            display.run()
            if not display.wait_key():
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv.destroyAllWindows()
