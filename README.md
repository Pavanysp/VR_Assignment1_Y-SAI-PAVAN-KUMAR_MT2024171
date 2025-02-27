# VR_Assignment1_Y-SAI-PAVAN-KUMAR_MT2024171

## Part 1: COINS DETECTION & SEGMENTATION & COUNTING USING OPENCV

### METHOD 1: CANNY EDGE DETECTOR

#### 1. Preprocessing the Images
- **Image Name:** `coins.jpg`
- Images are resized to **450x450** for consistency.
- **Converted to grayscale** using `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.
- **Gaussian Blur** `(7,7), 2` is applied to reduce noise and improve edge detection.

#### 2. Edge Detection
- **Canny edge detection** is applied using parameters:
  - `threshold1 = 50`
  - `threshold2 = 15`
- Two versions are created:
  - **Direct Canny Edge Detection**
  - **Canny Edge Detection with Histogram Equalization**

#### 3. Contour Detection and Filtering
- **Morphological closing** (`cv2.MORPH_CLOSE`) is used to remove small gaps in edges.
- Contours are detected using `cv2.findContours()`.
- Filtering is applied using area threshold: `cv2.contourArea(contour) > 500` to remove noise.

#### 4. Extracting and Counting Coins
- Coins are segmented using `cv2.bitwise_and()` and extracted based on bounding boxes.
- Each segmented coin is saved separately as `coin_{count}.jpg`.

#### OUTPUT IMAGES:
- `1_edges_c.jpg` (Direct Canny Edges)
- `2_contours_c.jpg` (Detected Contours)
- `coin_1_c.jpg, coin_2_c.jpg, ...` (Extracted Coins)

---

## Part 2: PANORAMA (KEY POINT DETECTION & IMAGE STITCHING)

### Project Steps:

#### 1. Image Preprocessing
- **Input Images:** `left.jpg, center.jpg, right.jpg`
- Images are resized to **500x500** for uniformity.

#### 2. Key Point Detection using SIFT
- **SIFT (Scale-Invariant Feature Transform)** is applied using `cv2.SIFT_create()`.
- Key points and descriptors are computed for all images.

#### 3. Feature Matching using Brute-Force Matcher
- **BFMatcher (Brute-Force Matcher)** is used for keypoint matching.
- **Lowe's Ratio Test** is applied to filter matches.

#### 4. Image Stitching using Homography & Blending
- **Homography matrices** are computed using **RANSAC**.
- Images are warped using perspective transformation.
- **Blending Strategy:** Uses `np.maximum()` to merge images smoothly.

#### OUTPUT IMAGES:
- `panorama.jpg` (Final Panorama Image)
- `matched1.jpg` (Matched Keypoints Left-Center)
- `matched2.jpg` (Matched Keypoints Right-Center)

---

## How to Run the Project

### Setup Requirements
Ensure OpenCV and NumPy are installed:
```sh
pip install opencv-python numpy
```

### Run Coin Detection
```sh
python coins.py
```

### Run Panorama Stitching
```sh
python panaroma.py
```

Ensure the **image paths** are updated in the script before running.

---

### Final Notes
- This project explores **coin detection** using the **Canny Edge Detector** in OpenCV.
- It also demonstrates **panorama stitching** using feature matching and homography.
- Make sure all **input images** are available before execution.

---

