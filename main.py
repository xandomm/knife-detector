import mediapipe as mp
import ctypes
import cv2

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
def display_alert(message):
    ctypes.windll.user32.MessageBoxW(0, message, "Alert", 0x40 | 0x1)  


def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    alert = False  # Flag to control when to display the alert
    if result.detections:
        for detection in result.detections:
            if detection.categories[0].category_name == "knife":
                print("Knife detected!")
                alert = True

    if alert:
        display_alert("CUIDADO, TEM ASSASSINO AE")

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='./ssd_mobilenet_v2.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    result_callback=print_result)

with ObjectDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detector.detect_async(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))

        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Webcam', bgr_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
