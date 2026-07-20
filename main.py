
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_blue = np.array([100, 150, 50])
upper_blue = np.array([130, 255, 255])

lower_orange = np.array([5, 150, 150])
upper_orange = np.array([20, 255, 255])

lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

colors = [
    ("BLUE", lower_blue, upper_blue, (255, 0, 0)),
    ("ORANGE", lower_orange, upper_orange, (0, 165, 255)),
    ("BLACK", lower_black, upper_black, (100, 100, 100))
]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for name, lower, upper, color in colors:

        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 500:

                x, y, w, h = cv2.boundingRect(cnt)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

    cv2.imshow("Color Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()