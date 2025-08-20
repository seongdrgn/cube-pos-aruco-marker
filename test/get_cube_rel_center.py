import cv2
import numpy as np
import d435cam
import time
from utils import *

def main():
    # --- 1. Initialization ---
    try:
        # Create camera object and define camera/lens parameters
        cam = d435cam.realsense_camera(720,1280,30)
        intrinsics = cam.get_intrinsics()
        cmtx = np.array([[intrinsics.fx,0.0,intrinsics.ppx],
                [0.0,intrinsics.fy,intrinsics.ppy],
                [0.0,0.0,1.0],])
        dist = np.array(intrinsics.coeffs)
    except Exception as e:
        print(f"Camera initialization error: {e}")
        return

    # Create ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # --- User Configuration ---
    REFERENCE_MARKER_ID = 0   # ID of the marker to be used as the reference coordinate system
    TARGET_MARKER_ID    = 1   # ID of the target marker
    MARKER_LENGTH_M     = 0.03  # Side length of the markers (in meters)

    # Define the offset from the target marker's center
    CUBE_SIDE_LENGTH = 0.066
    HALF_SIDE = CUBE_SIDE_LENGTH / 2.0
    # -------------------------

    # Define the 3D corner coordinates for the markers for solvePnP
    marker_3d_edges = np.array([
        [-MARKER_LENGTH_M/2, MARKER_LENGTH_M/2, 0], [MARKER_LENGTH_M/2, MARKER_LENGTH_M/2, 0],
        [MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0], [-MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0]
    ], dtype=np.float32)

    time.sleep(2.0)

    # --- 2. Main Loop ---
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret: break

        corners, ids, _ = detector.detectMarkers(frame)

        # Dictionary to store poses of only valid (correctly oriented) markers
        valid_poses = {}

        if ids is not None:
            for i, corner in enumerate(corners):
                ret_pnp, rvec, tvec = cv2.solvePnP(marker_3d_edges, corner, cmtx, dist)
                if ret_pnp:
                    # Z-axis directionality filter
                    R, _ = cv2.Rodrigues(rvec)
                    dot_product = np.dot(tvec.flatten(), R[:, 2])
                    if dot_product < 0:
                        valid_poses[ids[i][0]] = {'rvec': rvec, 'tvec': tvec, 'R': R}
                        cv2.drawFrameAxes(frame, cmtx, dist, rvec, tvec, MARKER_LENGTH_M * 0.8)

        # Proceed only if both the reference and target markers are validly detected
        if REFERENCE_MARKER_ID in valid_poses and TARGET_MARKER_ID in valid_poses:
            ref_pose = valid_poses[REFERENCE_MARKER_ID]
            target_pose = valid_poses[TARGET_MARKER_ID]

            R_ref, tvec_ref = ref_pose['R'], ref_pose['tvec']
            R_target, tvec_target = target_pose['R'], target_pose['tvec']

            # --- Core Logic: Calculate the offset point ---
            # 1. Define the offset in the TARGET marker's local coordinate system
            #    "coming down" the Z-axis means moving in the -Z direction.
            offset_local = np.array([[0], [0], [-HALF_SIDE]])

            # 2. Transform the local offset to the camera's coordinate system
            center_point_cam = np.dot(R_target, offset_local) + tvec_target

            # 3. Transform the camera-coordinate point to the REFERENCE marker's coordinate system
            relative_center_pos = np.dot(R_ref.T, center_point_cam - tvec_ref)
            # ---------------------------------------------
            
            # --- Visualization and Display ---
            # Display the final relative coordinates
            px, py, pz = relative_center_pos.flatten()
            text = f"Center @ Ref {REFERENCE_MARKER_ID}: ({px:.3f}, {py:.3f}, {pz:.3f}) m"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # For visualization, project the calculated 3D point onto the 2D image
            img_points, _ = cv2.projectPoints(center_point_cam, np.zeros(3), np.zeros(3), cmtx, dist)
            pt_2d = tuple(img_points[0][0].astype(int))
            cv2.circle(frame, pt_2d, 8, (0, 0, 255), -1) # Draw a red circle at the point

        cv2.imshow("Offset Point Tracker", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()