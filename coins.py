# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""Using Canny Edge detector"""

import cv2
import numpy as np

""" Only Canny Edge Detection"""
image = cv2.imread("C:/Users/hp/Downloads/coins.jpg")
image = cv2.resize(image, (450, 450))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 2)

edges = cv2.Canny(blurred, 50, 15)
cv2.imwrite("1_edges_c.jpg", edges)

kernel = np.ones((3, 3), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imwrite("2_contours_c.jpg", contour_image)

print(f"Total Coins Detected: {len(contours)}")

coin_count = 1
for contour in contours:
    if cv2.contourArea(contour) > 500: 
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        segmented_coin = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(contour)
        cropped_coin = segmented_coin[y:y+h, x:x+w]
        cv2.imwrite(f"coin_{coin_count}_c.jpg", cropped_coin)
        coin_count += 1

cv2.imshow("Edges", edges)
cv2.imshow("Contours", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""Canny-histogramequalisation"""
image = cv2.imread("C:/Users/hp/Downloads/coins.jpg")
image = cv2.resize(image, (450, 450))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)

blurred = cv2.GaussianBlur(equalized, (5, 5), 1)

edges = cv2.Canny(blurred, 30, 10)
cv2.imwrite("1_edges_h_c.jpg", edges)

kernel = np.ones((3, 3), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imwrite("2_contours_h_c.jpg", contour_image)

print(f"Total Coins Detected (Histogram Equalization and Canny): {len(contours)}")

coin_count = 1
for contour in contours:
    if cv2.contourArea(contour) > 500: 
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        segmented_coin = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(contour)
        cropped_coin = segmented_coin[y:y+h, x:x+w]
        cv2.imwrite(f"coin_h_c_{coin_count}.jpg", cropped_coin)
        coin_count += 1

cv2.imshow("Edges1", edges)
cv2.imshow("Contours1", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""Laplacian edge detection"""
image = cv2.imread("C:/Users/hp/Downloads/coins.jpg")
image = cv2.resize(image, (450, 450))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 2)

laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
edges = np.uint8(np.absolute(laplacian))

cv2.imwrite("1_edges_lap.jpg", edges)

kernel = np.ones((3, 3), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imwrite("2_contours_lap.jpg", contour_image)

print(f"Total Coins Detected using laplacian: {len(contours)}")

coin_count = 1
for contour in contours:
    if cv2.contourArea(contour) > 500: 
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        segmented_coin = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(contour)
        cropped_coin = segmented_coin[y:y+h, x:x+w]
        cv2.imwrite(f"coin_lap_{coin_count}.jpg", cropped_coin)
        coin_count += 1

cv2.imshow("Edges_Lap", edges)
cv2.imshow("Contours_Lap", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

