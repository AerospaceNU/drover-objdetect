import argparse
import cv2 as cv
from droverDetection.gui import Display

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the object detection application."
    )
    parser.add_argument(
        "--s",
        required=False,
        default=0,
        help="Source path reference for mp4. If none provided, defaults to live camera feed (camera 0).",
    )

    parser.add_argument(
        "--w",
        required=False,
        default="threshold.json",
        help="Weight reference for Detection module",
    )

    args = parser.parse_args()

    display = Display(args.s, windowName="Main", weights=args.w)

    try:
        while True:
            # Continues collecting frames from designated source until none.
            display.run()

            # If wait_key returns false (video ends), breaks main loop
            if not display.wait_key():
                break
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully if occurred
        pass
    finally:
        # Clean up and close all OpenCV windows for clean up.
        cv.destroyAllWindows()
