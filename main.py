from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import time
from pynput.mouse import Button, Controller

mouse = Controller()
def eye_aspect(eye):
	A, B, C = distance.euclidean(eye[1], eye[5]), distance.euclidean(
		eye[2], eye[4]), distance.euclidean(eye[0], eye[3])
	return (A + B) / (2.0 * C)


def eye_pos(eye):
	eye1, eye2 = eye[0], eye[3]
	return (int((eye1[0] + eye2[0]) / 2), int((eye1[1] + eye2[1]) / 2))


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

#Left eye
(lstart, lend) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
# right eye
(rstart, rend) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

threshold = .20
chk = 6

cap = cv2.VideoCapture(0)
flag = 0

while True:
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detector(gray, 0)

	for sub in subjects:
		shape = predictor(gray, sub)
		shape = face_utils.shape_to_np(shape)

		leftEye, rightEye = shape[lstart:lend], shape[rstart:rend]
		leftEar, rightEar = eye_aspect(leftEye), eye_aspect(rightEye)

		leftCenter, rightCenter = eye_pos(leftEye), eye_pos(rightEye)

		ear = (leftEar + rightEar) / 2.0

		leftEyeHull, rightEyeHull = cv2.convexHull(leftEye), cv2.convexHull(
			rightEye)

		# print(frame.shape,q leftCenter,rightCenter)
		cv2.circle(frame, eye_pos(leftEye), 2, (0, 255, 0), -1)
		# print(eye_pos(leftEye), mouse.position)
		mouse.position = eye_pos(leftEye)

		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < threshold:
			flag+=1
			if flag >= chk:
				mouse.press(Button.left)
				mouse.release(Button.left)
				break
		else:
			flag = 0

	cv2.namedWindow('Fra', cv2.WINDOW_NORMAL)
	cv2.imshow('Fra', frame)
	cv2.moveWindow('Fra', 1030, 0)
	key = cv2.waitKey(1) & 0xFF

	if key == ord('q'):
		break

cv2.destroyAllWindows()
