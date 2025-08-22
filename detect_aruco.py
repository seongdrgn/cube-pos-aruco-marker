import cv2
import numpy as np
import d435cam
import time

if __name__ == "__main__":
    cam = d435cam.realsense_camera(720,1280,30);
    intrinsics = cam.get_intrinsics();
    cmtx = [[intrinsics.fx,0.0,intrinsics.ppx],
            [0.0,intrinsics.fy,intrinsics.ppy],
            [0.0,0.0,1.0],];
    dist = intrinsics.coeffs;
    
    # board_type = cv2.aruco.DICT_6X6_250;
    board_type = cv2.aruco.DICT_4X4_50;
    arucoDict  = cv2.aruco.getPredefinedDictionary(board_type);
    parameters = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(arucoDict, parameters);
    
    time.sleep(2)
    
    # ret,img = cam.read()

    blue_BGR = (255, 0, 0)
    
    if(cam.isOpened()):
        while True:
            ret, frame = cam.read();
            if(ret):
                corners, ids, rejectedCandidates = detector.detectMarkers(frame)

                for corner in corners:
                    corner = np.array(corner).reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corner

                    topRightPoint    = (int(topRight[0]),      int(topRight[1]))
                    topLeftPoint     = (int(topLeft[0]),       int(topLeft[1]))
                    bottomRightPoint = (int(bottomRight[0]),   int(bottomRight[1]))
                    bottomLeftPoint  = (int(bottomLeft[0]),    int(bottomLeft[1]))

                    cv2.circle(frame, topLeftPoint, 4, blue_BGR, -1)
                    cv2.circle(frame, topRightPoint, 4, blue_BGR, -1)
                    cv2.circle(frame, bottomRightPoint, 4, blue_BGR, -1)
                    cv2.circle(frame, bottomLeftPoint, 4, blue_BGR, -1)
                
                cv2.imshow("img",frame);
                key=cv2.waitKey(1);
                if(key&0xff==ord('q')):
                    break

    cam.release();
    cv2.destroyAllWindows();
