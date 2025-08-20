import cv2
import numpy as np
import d435cam
import time
# from utils import *

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
    MARKER_LENGTH_M     = 0.01  # Side length of the markers (in meters)

    # Define the offset for the primary "center point"
    CUBE_SIDE_LENGTH = 0.067
    HALF_SIDE = CUBE_SIDE_LENGTH / 2.0
    
    # Define keypoint offsets from the calculated "center point"
    xy_offset = HALF_SIDE
    z_offset  = -0.0235 / 2.0
    keypoints_offset = np.array([
        [[ xy_offset], [ 0.0],       [z_offset]],
        [[-xy_offset], [ 0.0],       [z_offset]],
        [[ 0.0],       [ xy_offset], [z_offset]],
        [[ 0.0],       [-xy_offset], [z_offset]]
    ], dtype=np.float32)
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
                    # Get the top-left corner to position the text
                    topLeft = tuple(corner[0][0].astype(int))
                    marker_id_text = f"ID: {ids[i][0]}"
                    # Draw the ID text above the marker
                    cv2.putText(frame, marker_id_text, (topLeft[0], topLeft[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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

            # --- Core Logic Update ---
            
            # 1. Define the "center point" offset in the target marker's local frame
            center_point_offset_local = np.array([[0], [0], [-HALF_SIDE]])

            # 2. Calculate the camera-relative and reference-relative coordinates of the "center point"
            center_point_cam = np.dot(R_target, center_point_offset_local) + tvec_target
            relative_center_pos = np.dot(R_ref.T, center_point_cam - tvec_ref)

            # 3. Calculate the coordinates of the keypoints
            relative_keypoints_pos = []
            # for kp_offset in keypoints_offset:
            for i in range(4):
                kp_offset = keypoints_offset[i]

                # 3-1. Combine the center offset and keypoint offset in the target's local frame
                total_offset_local = center_point_offset_local + kp_offset
                
                # 3-2. Transform the combined local offset to the camera's coordinate system
                kp_cam = np.dot(R_target, total_offset_local) + tvec_target
                
                # 3-3. Transform the camera-coordinate keypoint to the reference marker's coordinate system
                kp_ref = np.dot(R_ref.T, kp_cam - tvec_ref)
                relative_keypoints_pos.append(kp_ref)
                
                # For visualization, project the camera-coordinate keypoint onto the image
                img_points, _ = cv2.projectPoints(kp_cam, np.zeros(3), np.zeros(3), cmtx, dist)
                pt_2d = tuple(img_points[0][0].astype(int))
                cv2.circle(frame, pt_2d, 6, (0, 0, 255), -1) # Keypoints in red

            # --- Visualization and Display ---
            # Visualize the "center point" itself
            center_img_points, _ = cv2.projectPoints(center_point_cam, np.zeros(3), np.zeros(3), cmtx, dist)
            center_pt_2d = tuple(center_img_points[0][0].astype(int))
            cv2.circle(frame, center_pt_2d, 8, (255, 255, 255), -1) # Center point in white

            # Display the final relative coordinates as text
            print(f"Relative Center Position: {relative_center_pos.flatten()}")
            px, py, pz = relative_center_pos.flatten()
            center_text = f"Center @ Ref {REFERENCE_MARKER_ID}: ({px:.3f}, {py:.3f}, {pz:.3f}) m"
            cv2.putText(frame, center_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            for i, kp_pos in enumerate(relative_keypoints_pos):
                kpx, kpy, kpz = kp_pos.flatten()
                kp_text = f"KP{i+1}: ({kpx:.3f}, {kpy:.3f}, {kpz:.3f}) m"
                cv2.putText(frame, kp_text, (20, 70 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)

        cv2.imshow("Keypoint Tracker", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()