import cv2 as cv
import numpy as np

# Load images
left = cv.imread("C:/Users/hp/Downloads/left.jpg")
center = cv.imread("C:/Users/hp/Downloads/center.jpg")
right = cv.imread("C:/Users/hp/Downloads/right.jpg")

left = cv.resize(left, (500, 500))
center = cv.resize(center, (500, 500))
right = cv.resize(right, (500, 500))

# Convert to grayscale
gray1 = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(center, cv.COLOR_BGR2GRAY)
gray3 = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

# Detect SIFT keypoints and descriptors
sift = cv.SIFT_create()
keyp1, des1 = sift.detectAndCompute(gray1, None)
keyp2, des2 = sift.detectAndCompute(gray2, None)
keyp3, des3 = sift.detectAndCompute(gray3, None)

# Draw keypoints
img1_sift = cv.drawKeypoints(left, keyp1, None, color=(255, 0, 0))
img2_sift = cv.drawKeypoints(center, keyp2, None, color=(255, 0, 0))
img3_sift = cv.drawKeypoints(right, keyp3, None, color=(255, 0, 0))

cv.imwrite("C:/Users/hp/Downloads/1_sift.jpg", img1_sift)
cv.imwrite("C:/Users/hp/Downloads/2_sift.jpg", img2_sift)
cv.imwrite("C:/Users/hp/Downloads/3_sift.jpg", img3_sift)

# Feature Matching
bf = cv.BFMatcher()
matches12 = bf.knnMatch(des1, des2, k=2)
matches32 = bf.knnMatch(des3, des2, k=2)

good_matches12 = [m for m, n in matches12 if m.distance < 0.75 * n.distance]
good_matches32 = [m for m, n in matches32 if m.distance < 0.75 * n.distance]

matching_img12 = cv.drawMatches(left, keyp1, center, keyp2, good_matches12, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
matching_img32 = cv.drawMatches(right, keyp3, center, keyp2, good_matches32, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow("Matches Left-Center", matching_img12)
cv.imshow("Matches Right-Center", matching_img32)
cv.imwrite("C:/Users/hp/Downloads/matched1.jpg", matching_img12)
cv.imwrite("C:/Users/hp/Downloads/matched2.jpg", matching_img32)

# Function to get matched points
def get_matched_points(good_matches, kpA, kpB):
    ptsA = np.float32([kpA[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return ptsA, ptsB

pts1, pts2 = get_matched_points(good_matches12, keyp1, keyp2)  # Left to Center
pts3, pts2_ = get_matched_points(good_matches32, keyp3, keyp2)  # Right to Center

H12, _ = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0) if len(pts1) >= 4 else None
H32, _ = cv.findHomography(pts3, pts2_, cv.RANSAC, 5.0) if len(pts3) >= 4 else None

if H12 is None or H32 is None:
    print("Error: Not enough keypoints matched!")
    exit()

# Warp images
h, w = center.shape[:2]
corners = np.array([[[0, 0]], [[0, h]], [[w, 0]], [[w, h]]], dtype=np.float32)
warped_corners1 = cv.perspectiveTransform(corners, H12)
warped_corners3 = cv.perspectiveTransform(corners, H32)

x_min = min(warped_corners1[:, 0, 0].min(), 0, warped_corners3[:, 0, 0].min())
x_max = max(warped_corners1[:, 0, 0].max(), w, warped_corners3[:, 0, 0].max())
y_min = min(warped_corners1[:, 0, 1].min(), 0, warped_corners3[:, 0, 1].min())
y_max = max(warped_corners1[:, 0, 1].max(), h, warped_corners3[:, 0, 1].max())

offset_x = int(abs(x_min))
offset_y = int(abs(y_min))

panorama2_width = int(x_max - x_min)
panorama2_height = int(y_max - y_min)

translation_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

H12 = translation_matrix @ H12 
H32 = translation_matrix @ H32  

left_warped = cv.warpPerspective(left, H12, (panorama2_width, panorama2_height))
right_warped = cv.warpPerspective(right, H32, (panorama2_width, panorama2_height))
center_warped = cv.warpPerspective(center, translation_matrix, (panorama2_width, panorama2_height))

panorama = np.maximum(np.maximum(left_warped, center_warped), right_warped)
cv.imshow("Panorama", panorama)
cv.imshow("Panorama", panorama)
cv.imwrite("C:/Users/hp/Downloads/panorama.jpg", panorama)
print("Panorama saved as C:/Users/hp/Downloads/panorama.jpg")
