# import necessary libraries/modules:
import numpy as np
import cv2
import  imutils
import sys
import pytesseract
import pandas as pd
import time

# Read the input image car_2.jpg using OpenCV:
img = cv2.imread('car_2.jpg')

# Resize the image to a width of 500 pixels using imutils:
img = imutils.resize(img, width=500)

# Display the original image using OpenCV's cv2.imshow() function:
cv2.imshow("ORIGINAL IMAGE: ", img)

# Convert the image to grayscale:
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("1 - Grayscale Conversion", gray)

# Apply bilateral filter to reduce noise while preserving edges:
gray = cv2.bilateralFilter(gray, 11, 17, 17)
#cv2.imshow("2 - Bilateral Filter", gray)

# Detect edges in the image using Canny edge detection:
edged = cv2.Canny(gray, 170, 200)
#cv2.imshow("4 - Canny Edges", edged)

# Find contours in the edged image:
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by area in descending order and select the largest 30 contours:
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
NumberPlateCnt = None 

count = 0
# Loop through each contour and approximate the contour shape to a polygon with four vertices 
# (presumably representing the number plate):
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  
            NumberPlateCnt = approx 
            break

# Create a mask to extract the region of interest (number plate) from the image:
# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_img = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_img = cv2.bitwise_and(img,img,mask=mask)
cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image",new_img)

# Configuration for tesseract
config = ('-l eng --oem 1 --psm 3')

# Run tesseract OCR on image
text = pytesseract.image_to_string(new_img, config=config)

#Data is stored in CSV file (Pandas DataFrame)
raw_data = {'date': [time.asctime( time.localtime(time.time()) )], 
        'v_number': [text]}
df = pd.DataFrame(raw_data, columns = ['date', 'v_number'])

# Save the DataFrame to a CSV file named data.csv:
df.to_csv('data.csv')

# Print recognized text
print(text)

# Wait for a key press to close the displayed images:
cv2.waitKey(0)
