import cv2
import numpy as np
num_vid = [4]
for i in num_vid:

    def define_ROI(event, x, y, flags, param):
        global r, c, w, h, roi_defined
        # if the left mouse button was clicked,
        # record the starting ROI coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            r, c = x, y
            roi_defined = False
        # if the left mouse button was released,
        # record the ROI coordinates and dimensions
        elif event == cv2.EVENT_LBUTTONUP:
            r2, c2 = x, y
            h = abs(r2 - r)
            w = abs(c2 - c)
            r = min(r, r2)
            c = min(c, c2)
            roi_defined = True


    roi_defined = False

    cap = cv2.VideoCapture(f"Data/eye{i}.mp4")
    ret, frame = cap.read()
    clone = frame.copy()
    cv2.namedWindow("First image")
    cv2.setMouseCallback("First image", define_ROI)
    while True:
        # display the image and wait for a keypress
        cv2.imshow("First image", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the ROI is defined, draw it!
        if (roi_defined):
            # draw a green rectangle around the region of interest
            cv2.rectangle(frame, (r, c), (r + h, c + w), (0, 255, 0), 2)
        # else reset the image...
        else:
            frame = clone.copy()
        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break
    while True:

        ret, frame = cap.read()
        if ret is False:
            break
        if roi_defined:
            roi = frame[c: c+w, r: r+h]
            rows, cols, _ = roi.shape
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
            _, threshold = cv2.threshold(gray_roi, 20, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
            # contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            for cnt in contours:

                cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
                x, y, wb, hb = cv2.boundingRect(cnt)
                cv2.rectangle(roi, (x, y), (x + wb, y + hb), (255, 0, 0), 2)
                cv2.line(roi, (x + int(wb/2), 0), (x + int(wb/2), rows), (0, 255, 0), 2)
                cv2.line(roi, (0, y + int(hb/2)), (cols, y + int(hb/2)), (0, 255, 0), 2)
                break
            cv2.imshow("Threshold", threshold)
            cv2.imshow("gray roi", gray_roi)
            cv2.imshow("Roi", roi)
            key = cv2.waitKey(30)
            if key == 27:
                break
            if key == ord('p'):
                cv2.waitKey(-1)
    cv2.destroyAllWindows()
    # gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    # _, threshold = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY_INV)
