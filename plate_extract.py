#!/usr/bin/env python3
"""
plate_extract.py - Vehicle Number Plate Recognition using OpenCV and Tesseract OCR.
"""

import argparse
import sys
import time

import cv2
import imutils
import numpy as np
import pandas as pd
import pytesseract
import re


def main():
    parser = argparse.ArgumentParser(
        description="ANPR: Extract and OCR the vehicle number plate from an image."
    )
    parser.add_argument(
        "-i", "--image", required=True, help="Path to the input image"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data.csv",
        help="Path to the output CSV file (default: data.csv)",
    )
    parser.add_argument(
        "--psm",
        default="3",
        help="Tesseract Page Segmentation Mode (PSM) (default: 3)",
    )
    parser.add_argument(
        "--oem",
        default="1",
        help="Tesseract OCR Engine Mode (OEM) (default: 1)",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract language (default: eng)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not display intermediate images in windows",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not read image '{args.image}'", file=sys.stderr)
        sys.exit(1)
    img = imutils.resize(img, width=500)
    if not args.no_display:
        cv2.imshow("Original Image", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 170, 200)

    cnts, _ = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    plate_contour = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is None:
        print("Error: no number plate contour detected", file=sys.stderr)
        sys.exit(1)

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], -1, 255, -1)
    plate_img = cv2.bitwise_and(img, img, mask=mask)
    if not args.no_display:
        cv2.namedWindow("Plate Region", cv2.WINDOW_NORMAL)
        cv2.imshow("Plate Region", plate_img)

    config = f"-l {args.lang} --oem {args.oem} --psm {args.psm}"
    text = pytesseract.image_to_string(plate_img, config=config)
    # Clean OCR output: keep only alphanumeric characters (letters and digits)
    text = re.sub(r'[^A-Za-z0-9]', '', text)

    raw_data = {
        "date": [time.asctime(time.localtime(time.time()))],
        "v_number": [text],
    }
    df = pd.DataFrame(raw_data, columns=["date", "v_number"])
    df.to_csv(args.output, index=False)

    print(text.strip())
    if not args.no_display:
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
