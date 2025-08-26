import cv2
import pytesseract
import numpy as np

# OPTIONAL: Set tesseract path for Windows users
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return None

    print("Press 'S' to capture image or 'Q' to exit.")
    img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        cv2.imshow("Camera - Press 'S' to Capture, 'Q' to Quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('s'):
            img = frame.copy()
            print("Image captured.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return img

def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize for better OCR accuracy
    scale_percent