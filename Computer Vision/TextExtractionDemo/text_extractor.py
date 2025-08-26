import cv2
import pytesseract
import numpy as np

# Set up Tesseract path (update if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = 'sample1.jpg'  # Replace with your image name if different
img = cv2.imread(image_path)

# Step 1: Preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 2: Contour Detection to Draw Boxes
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 50 and h > 15:  # filter small areas
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Step 3: OCR Text Extraction
custom_config = r'--oem 3 --psm 6'  # config for better OCR results
extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

# Step 4: Save Outputs
cv2.imwrite("output_with_boxes.jpg", img)
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)

# Step 5: Display Results
print("=== Extracted Text ===\n")
print(extracted_text)
print("\nText saved to 'extracted_text.txt'")
print("Annotated image saved as 'output_with_boxes.jpg'")

# Optional: Show the output image
cv2.imshow("Detected Text", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
