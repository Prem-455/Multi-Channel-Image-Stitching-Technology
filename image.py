import cv2 
import numpy as np

def load_images(image_paths)
    {image_paths} = ['image1.jpg', 'image2.jpg', 'image3.jpg']  
    images = [ cv2.imread(path) for path in image_paths]
    images = [cv2.resize(img, (800, 600)) for img in images] 
    return images

def detect_and_match_features(images):
    sift = cv2.SIFT_create()
    keypoints, descriptors = [], 
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = [bf.match(descriptors[i], descriptors[i + 1]) for i in range(len(descriptors) - 1)]
    matches = [sorted(m, key=lambda x: x.distance) for m in matches]
    return keypoints, matches


def stitch_images(images, keypoints, matches):
    stitched = images[0]
    for i in range(len(matches)):
        src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in matches[i]]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches[i]]).reshape(-1, 1, 2)
        
      
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
      
        h, w = images[i + 1].shape[:2]
        warped = cv2.warpPerspective(images[i + 1], H, (stitched.shape[1] + w, stitched.shape[0]))
        
        
        stitched = blend_images(stitched, warped)
    return stitched

def blend_images(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    blended = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    blended[:h1, :w1] = img1
    blended[:h2, :w2] = cv2.addWeighted(img2[:h1, :w1], 0.5, blended[:h1, :w1], 0.5, 0)
    return blended

def cylindrical_projection(img):
    h, w = img.shape[:2]
    K = np.array([[w / 2, 0, w / 2], [0, w / 2, h / 2], [0, 0, 1]])
    return cv2.warpPerspective(img, K, (w, h), flags=cv2.WARP_INVERSE_MAP)

if __name__ == "__main__":

    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']  
    images = load_images(image_paths)

    keypoints, matches = detect_and_match_features(images)
    
    stitched_result = stitch_images(images, keypoints, matches)
  
    stitched_360 = cylindrical_projection(stitched_result)
    

    cv2.imshow("Stitched Image", stitched_result)
    cv2.imshow("360° View", stitched_360)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite("stitched_result.jpg", stitched_result)
    cv2.imwrite("stitched_360.jpg", stitched_360)
