import cv2
import pytesseract
import numpy as np
from tkinter import filedialog, Tk

# Set Tesseract OCR executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# GUI File dialog to select image
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select an image",
                                       filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
if not file_path:
    print("No file selected. Exiting.")
    exit()

# Load image
image = cv2.imread(file_path)

# Resize large image down to avoid blurring
max_height = 1000
if image.shape[0] > max_height:
    scale = max_height / image.shape[0]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# 1. Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Apply bilateral filter (preserves edges, reduces noise)
filtered = cv2.bilateralFilter(gray, 11, 17, 17)

# 3. Adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(filtered, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 10)

# 4. Morphological Opening (remove small noise)
kernel = np.ones((1, 1), np.uint8)
opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

# 5. Sharpening filter to enhance text
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(opened, -1, sharpen_kernel)

# Optional: Resize for better OCR if text is small
final = cv2.resize(sharpened, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# 6. OCR with improved config
custom_config = r'--oem 3 --psm 6 -l eng'
extracted_text = pytesseract.image_to_string(final, config=custom_config)

# 7. Save results
output_img = image.copy()
cv2.imwrite("high_accuracy_boxes.jpg", output_img)
with open("high_accuracy_text.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)

# 8. Show output
print("✅ OCR Complete — High Accuracy Mode")
print("📝 Extracted Text saved to 'high_accuracy_text.txt'")
print("📸 Input Image saved as 'high_accuracy_boxes.jpg'")
print("\n=== Extracted Text ===\n")
print(extracted_text)

# Optional: Show processed image
cv2.imshow("Processed Image", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
