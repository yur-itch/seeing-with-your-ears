import cv2
import image2sound

def preview_prepared_video(video_path, hilbert_iterations, window_size=512):
    """
    Demonstrate how the video frames look after preparation (cropping, resizing, grayscale).

    Closes correctly when the user closes the window.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    window_name = "Prepared Frame Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare frame using image2sound
        img_prepared = image2sound.load_and_prepare_image_from_array(frame, hilbert_iterations)

        # Resize for display
        display_img = cv2.resize(img_prepared, (window_size, window_size))
        cv2.imshow(window_name, display_img)

        # Wait 30 ms
        key = cv2.waitKey(30) & 0xFF

        # Exit on any key press
        if key != 255:
            break

        # Exit if window was closed manually
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "samples/videos/bouncing_ball.mp4"
    hilbert_iterations = 6
    preview_prepared_video(video_path, hilbert_iterations)