import cv2
import numpy as np


def find_obj(frame, chan_1, chan_2): 
    captured_frame = frame.copy()

    # Convert original image to BGR
    captured_frame_bgr = cv2.cvtColor(captured_frame, cv2.COLOR_BGRA2BGR)
    # First blur to reduce noise prior to color space conversion
    captured_frame_bgr = cv2.medianBlur(captured_frame_bgr, 3)
    # Convert to Lab color space, check r(a-channel)
    captured_frame_lab = cv2.cvtColor(captured_frame_bgr, cv2.COLOR_BGR2Lab)
    # Threshold for red
    captured_frame_lab_red = cv2.inRange(captured_frame_lab, np.array([20, 150, 150]), np.array([190, 255, 255]))
    # Second blur to reduce more noise, easier circle detection
    captured_frame_lab_red = cv2.GaussianBlur(captured_frame_lab_red, (5, 5), 2, 2)
    # Use the Hough transform to detect circles in the image
    circles = cv2.HoughCircles(captured_frame_lab_red, cv2.HOUGH_GRADIENT, 1, captured_frame_lab_red.shape[0] / 8, param1=100, param2=18, minRadius=5, maxRadius=60)

	# If we have extracted a circle, draw an outline of one circle
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        cv2.circle(frame, center=(circles[0, 0], circles[0, 1]), radius=circles[0, 2], color=(0, 0, 255), thickness=2)
        return circles[0,0], circles[0,1]

    '''
        my code
    '''
    # filtered = cv2.subtract(chan_1, chan_2) 
    # blur = cv2.medianBlur(filtered,15)

    # ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)

    # # get all contours
    # cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # most_circular_contour = None
    # max_circularity = 0

    # for contour in cnts:
    #     perimeter = cv2.arcLength(contour, True)
    #     if perimeter == 0: continue
    #     area = cv2.contourArea(contour)
        
    #     # Calculate circularity
    #     circularity = 4 * np.pi * area / (perimeter ** 2)
        
    #     # Check if contour is more circular than the previous one
    #     if circularity > max_circularity:
    #         max_circularity = circularity
    #         most_circular_contour = contour

    # find max contour and assume to be the object
    # if len(cnts) > 0:
    #     x, y, w, h = cv2.boundingRect(circles)
        
    #     x_coord = (x + w) / 2
    #     b_mask = np.zeros(frame.shape)
    #     b_mask = cv2.drawContours(b_mask, cnts, 0, 255, cv2.FILLED) #get biggest contour

    #     # return the mask of the object(demo display) and the x-coordinate of the middle of the object for bit-processing
    #     return x_coord, b_mask
    
    return 0, 0

if __name__ == '__main__':

    vid = cv2.VideoCapture(0) # 'http://172.20.10.1'

    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # dim = (256, 256)
        # frame = cv2.resize(frame, dim)
        #print(frame.shape)
        b,g,r = cv2.split(frame)
        chan1, chan2 = b, g

        x, y = find_obj(frame, chan1, chan2)
        print(x, y)
        cv2.imshow("img", frame)

        # Display the resulting frame
        # if isinstance(mask, int):
        #     cv2.imshow("img", frame)
        # elif (mask.any() == 0):
        #     cv2.imshow("img", frame)
        # else:
        #     cv2.imshow("img", mask)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
