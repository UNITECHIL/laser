import cv2
import numpy as np
import dlib

def get_eyes(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    face = faces[0]
    shape = predictor(gray, face)

    left_eye_indices = [36, 37, 38, 39, 40, 41]
    right_eye_indices = [42, 43, 44, 45, 46, 47]

    left_eye_pts = np.array([[shape.part(i).x, shape.part(i).y] for i in left_eye_indices], dtype=int)
    right_eye_pts = np.array([[shape.part(i).x, shape.part(i).y] for i in right_eye_indices], dtype=int)

    return left_eye_pts, right_eye_pts, img

def replace_eyes(image_path, eye_image_path):
    left_eye_pts, right_eye_pts, img = get_eyes(image_path)

    eye_image = cv2.imread(eye_image_path)
    eye_width = right_eye_pts[:, 0].max() - left_eye_pts[:, 0].min()
    eye_height = right_eye_pts[:, 1].max() - right_eye_pts[:, 1].min()

    for eye_pts in [left_eye_pts, right_eye_pts]:
        # Resize the eye image to match the width of the eyes
        eye_image_resized = cv2.resize(eye_image, (eye_width, eye_height))

        # Calculate the mean position of the eyes
        mean_x = int(eye_pts[:, 0].mean())
        mean_y = int(eye_pts[:, 1].mean())

        # Calculate the region to place the eye image
        roi_x = max(0, mean_x - eye_width // 2)
        roi_y = max(0, mean_y - eye_height // 2)
        roi_width = min(img.shape[1] - roi_x, eye_width)
        roi_height = min(img.shape[0] - roi_y, eye_height)

        # Resize the eye image to the calculated region
        eye_image_roi = cv2.resize(eye_image_resized, (roi_width, roi_height))

        # Create a mask with the eye region
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.fillPoly(mask, [eye_pts], (255, 255, 255))

        # Blend the eye image with the original image
        img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = cv2.addWeighted(
            img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width], 1 - 0.5,
            eye_image_roi, 0.5, 0
        )

    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "/users/home/desktop/boteyes/1.png"
    eye_image_path = "/users/home/desktop/boteyes/eye_image.png"

    replace_eyes(image_path, eye_image_path)
