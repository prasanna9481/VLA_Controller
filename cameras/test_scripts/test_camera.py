"""
Purpose:
Tests a specific ZED camera by opening it with its serial number,
displaying a live stream, and saving one captured frame.

Workflow:
1. Create and configure the ZED camera for HD720 at 30 FPS.
2. Select the physical camera using its serial number.
3. Continuously grab frames from the LEFT camera view.
4. Convert the ZED BGRA image into a BGR NumPy array for OpenCV.
5. Display the live stream in an OpenCV window.
6. Save the first successfully captured frame as `zed_frame.png`.
7. Stop when `q` is pressed and close the camera cleanly.

Technical details:
- Camera selection is done with `set_from_serial_number()`.
- Frames are retrieved using `sl.VIEW.LEFT`.
- ZED returns BGRA image data; the alpha channel is removed for OpenCV.
- OpenCV uses BGR color ordering for display and image saving.
"""



import cv2
import pyzed.sl as sl


def main():
    # Create ZED camera object
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    # Select a specific ZED camera
    init_params.set_from_serial_number(39337350)

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {status}")
        return

    print("ZED camera opened successfully")

    # Container where ZED will put the image
    image = sl.Mat()

    saved = False

    print("Press 'q' to quit")

    while True:
        # Grab a new frame
        if zed.grab() == sl.ERROR_CODE.SUCCESS:

            # Retrieve LEFT camera image
            zed.retrieve_image(image, sl.VIEW.LEFT)

            # Convert ZED Mat -> NumPy array
            frame = image.get_data()

            # ZED usually returns BGRA.
            # OpenCV only needs BGR for display/save.
            frame_bgr = frame[:, :, :3]

            # Show live camera
            cv2.imshow("ZED Left Camera", frame_bgr)

            # Save one image automatically
            if not saved:
                cv2.imwrite("zed_frame.png", frame_bgr)
                print("Saved image: zed_frame.png")
                saved = True

        # q = quit
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # Cleanup
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()