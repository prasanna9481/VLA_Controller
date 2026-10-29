#!/usr/bin/env python3

import cv2
from pathlib import Path
from cameras.zed_camera import DualZEDCameras

SAVE_DIR = Path("robot/goal_images")


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    cameras = DualZEDCameras()

    try:
        print("Press 's' to save goal images, 'q' to quit.")

        while True:
            external_rgb, wrist_rgb = cameras.get_frames_rgb()

            cv2.imshow("External Camera", cv2.cvtColor(external_rgb, cv2.COLOR_RGB2BGR))
            cv2.imshow("Wrist Camera", cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                cv2.imwrite(str(SAVE_DIR / "external_goal.png"), cv2.cvtColor(external_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(SAVE_DIR / "wrist_goal.png"), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
                print(f"Goal images saved to {SAVE_DIR.resolve()}")

            elif key == ord("q"):
                break

    finally:
        cameras.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()