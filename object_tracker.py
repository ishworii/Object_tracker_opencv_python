import cv2
import numpy as np
from converter import convert
import imutils


def write_text(frame, bottomLeftCornerOfText, s, fontScale=1):
    font = cv2.FONT_HERSHEY_PLAIN
    fontColor = (0, 0, 255)
    lineType = 1

    cv2.putText(frame, s,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


cap = cv2.VideoCapture(0)

# lower and upper limit for the mask for a yellow ball
lower_range, upper_range = convert(93, 202, 216)


while True:
    ret, frame = cap.read()
    # flip the frames
    frame = cv2.flip(frame, 1)
    # blur the frame
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialise the current
    # (x,y) center of the ball

    cnts = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only process if atleast one contour was found
    if len(cnts) > 0:
        # draw only the largest contour or draw all
        c = max(cnts, key=cv2.contourArea)
        # for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # only proceed if the radius meets a minimum size
        if radius > 20:
            #  draw the circle and centroid on the frame,
            #  then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            write_text(frame, (10, 30), 'Object Detected', 2)
            write_text(frame, (int(x), int(y) - int(radius)),
                       f'X:{int(x)},Y:{int(y)}')
            # draw a line from the center of the frame to the center of the contour
            (h, w) = frame.shape[:2]
            cv2.line(frame, (h // 2, w // 2),
                     (int(x), int(y)), (0, 10, 255), 1, 4)
    else:
        write_text(frame, (10, 30), 'Object Not Detected', 2)
        write_text(frame, (10, 70), 'Recalibrating...', 2)

    cv2.imshow('live mask', mask)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
