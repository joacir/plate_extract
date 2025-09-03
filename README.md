## Vehicle Number Plate Recognition using CV2

This Python script demonstrates an Automatic Number Plate Recognition (ANPR) system using OpenCV and Tesseract OCR. It detects and extracts the number plate from an input image and recognizes the characters on the plate.

### Dependencies:

- **OpenCV**: A popular computer vision library used for image processing tasks such as reading, resizing, filtering, edge detection, contour detection, and drawing.
- **NumPy**: A fundamental package for scientific computing with Python, used here for array manipulation.
- **imutils**: A package providing convenience functions to make basic image processing functions such as resizing, rotating, and displaying images easier with OpenCV.
- **pytesseract**: A Python wrapper for Google's Tesseract-OCR Engine. It allows for easy integration of OCR capabilities into Python applications.
- **Pandas**: A powerful data manipulation and analysis library. Here it's used to store the recognized text and current timestamp into a CSV file.
- **System Libraries**: On Linux, ensure that the OpenGL shared object (libGL.so.1) is installed. For example, on Ubuntu/Debian:
  ```bash
  sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
  ```

### Functionality:

1. **Read Image**: The script reads the input image 'car_2.jpg' using OpenCV.

2. **Preprocessing**:
   - Resize the image to a width of 500 pixels.
   - Convert the image to grayscale.
   - Apply bilateral filtering to reduce noise while preserving edges.
   - Detect edges using Canny edge detection.

3. **Contour Detection**:
   - Find contours in the edged image.
   - Sort contours by area in descending order and select the largest 30 contours.
   - Loop through each contour and approximate the shape to a polygon with four vertices, presumably representing the number plate.

4. **Number Plate Extraction**:
   - Create a mask to extract the region of interest (number plate) from the image.
   - Apply the mask to the original image.

5. **Text Recognition**:
   - Configure Tesseract OCR.
   - Run OCR on the extracted number plate region to recognize the text.

6. **OCR Output Cleaning**:
   - Apply a regular expression to retain only alphanumeric characters (letters and digits), removing any other characters and whitespace.

7. **Data Storage**:
   - Store the recognized text along with the current timestamp in a Pandas DataFrame.

8. **Output**:
   - Display the final image with the extracted number plate region.
   - Print the recognized text.
   - Save the DataFrame to a CSV file named 'data.csv'.

### How to Use:

1. Make sure you have Python 3 installed along with the necessary libraries: OpenCV, NumPy, imutils, pytesseract, and pandas.
2. Clone the repository or download the `plate_extract.py` script.
3. Run the script by passing the image as an argument:
   ```bash
   python plate_extract.py -i path/to/image.jpg
   ```
4. (Optional) Use `--psm`, `--oem`, `--lang`, `-o` and `--no-display` to adjust OCR settings, output CSV file, and display mode.
5. The recognized plate text will be printed to the console, and the CSV file will be saved as specified by the `-o` parameter (default: data.csv).

### Note:

- Ensure that Tesseract OCR is properly installed on your system and its executable is in the PATH.
- Adjust the parameters and configurations as needed for different input images and desired performance.

### Code Reference Notice:

- The code provided in this repository was referenced from [Vehicle-Number-Plate-Reading](https://github.com/vjgpt/Vehicle-Number-Plate-Reading) and was thoroughly understood before being rewritten.
- While the core functionality remains the same, modifications have been made for clarity.
- All credit for the original implementation goes to the author(s) of the referenced repository.
