import cv2
import pytesseract
import numpy as np

# Set up path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

print("Press 's' to scan & extract text from current frame.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Step 1: Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Threshold to black & white
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Morphological processing (dilate)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Step 5: Contour detection to show boxes
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxed_frame = frame.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 80 and h > 20:
            cv2.rectangle(boxed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show live preview with boxes
    cv2.imshow("Live OCR - Press 's' to Scan", boxed_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print("\n[INFO] Scanning and extracting text...")

        # Resize image for better OCR accuracy
        resized = cv2.resize(dilated, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        # Tesseract OCR config
        custom_config = r'--oem 3 --psm 11 -l eng'

        # Optional: use whitelist (uncomment to enable)
        # custom_config += r' -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

        # Perform OCR
        extracted_text = pytesseract.image_to_string(resized, config=custom_config)

        # Show result
        print("=== Extracted Text ===\n")
        print(extracted_text)

        # Save snapshot and text
        cv2.imwrite("realtime_capture.jpg", boxed_frame)
        with open("realtime_extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(extracted_text)

        print("\n[Saved] realtime_capture.jpg")
        print("[Saved] realtime_extracted_text.txt")

    elif key == ord('q'):
        print("[INFO] Quitting...")
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()
