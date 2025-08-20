import cv2
import numpy as np
import d435cam
import time
from utils import *

if __name__ == "__main__":
    # 카메라 생성 및 카메라, 렌즈 파라메터 정의
    cam = d435cam.realsense_camera(720,1280,30)
    intrinsics = cam.get_intrinsics()
    cmtx = np.array([[intrinsics.fx,0.0,intrinsics.ppx],
            [0.0,intrinsics.fy,intrinsics.ppy],
            [0.0,0.0,1.0],])
    dist = np.array(intrinsics.coeffs)
    
    # aruco detector 생성
    # board_type = cv2.aruco.DICT_6X6_250;
    board_type = cv2.aruco.DICT_4X4_50
    arucoDict  = cv2.aruco.getPredefinedDictionary(board_type);
    parameters = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(arucoDict, parameters);
    
    #realsense 카메라 초기 노출시간 확보
    time.sleep(2)
    
    # length of the cube
    CUBE_SIDE_LENGTH = 0.065
    HALF_SIDE = CUBE_SIDE_LENGTH / 2.0

    # length of the marker side in meters
    MARKER_LENGTH_M = 0.01
    marker_3d_edges = np.array([
        [-MARKER_LENGTH_M / 2, MARKER_LENGTH_M / 2, 0],
        [MARKER_LENGTH_M / 2, MARKER_LENGTH_M / 2, 0],
        [MARKER_LENGTH_M / 2, -MARKER_LENGTH_M / 2, 0],
        [-MARKER_LENGTH_M / 2, -MARKER_LENGTH_M / 2, 0]
    ], dtype=np.float32)
    
    # cube marker index
    CUBE_MARKER_IDS = {1, 2, 3, 4, 5, 6}

    #파란색상 정의
    blue_BGR = (255, 0, 0)
    green_BGR = (0, 255, 0)
    red_BGR = (0, 0, 255)
    
    if(cam.isOpened()):
        while True:
            ret, frame = cam.read()
            if not ret: break

            corners, ids, _ = detector.detectMarkers(frame)

            if ids is not None:
                # 모든 보이는 마커의 Pose를 계산하여 저장
                poses = {}
                for i, corner in enumerate(corners):
                    ret_pnp, rvec, tvec = cv2.solvePnP(marker_3d_edges, corner, cmtx, dist)
                    if ret_pnp:
                        poses[ids[i][0]] = {'rvec': rvec, 'tvec': tvec}
                        # (옵션) 모든 마커의 좌표축을 그려서 확인
                        cv2.drawFrameAxes(frame, cmtx, dist, rvec, tvec, MARKER_LENGTH_M * 0.8)

                # 큐브 중심점 후보들을 저장할 리스트
                estimated_cube_centers_cam = []

                # 각 큐브 마커로부터 중심점 후보 계산
                for marker_id in poses:
                    if marker_id in CUBE_MARKER_IDS:
                        rvec_face = poses[marker_id]['rvec']
                        tvec_face = poses[marker_id]['tvec']
                        R_face, _ = cv2.Rodrigues(rvec_face)

                        # 로컬 -Z 방향으로 오프셋 벡터 생성
                        offset_local = np.array([[0], [0], [-HALF_SIDE]])
                        # 카메라 좌표계로 오프셋 변환
                        offset_cam = np.dot(R_face, offset_local)
                        # 중심점 후보 계산
                        center_candidate = tvec_face + offset_cam
                        estimated_cube_centers_cam.append(center_candidate)

                # 중심점 후보가 하나라도 있으면 평균을 계산하여 최종 중심으로 결정
                if estimated_cube_centers_cam:
                    final_cube_center_cam = np.mean(estimated_cube_centers_cam, axis=0)

                    # 기준 마커(ID 0)가 보이면 상대 좌표 계산
                    if 0 in poses:
                        rvec_ref = poses[0]['rvec']
                        tvec_ref = poses[0]['tvec']
                        R_ref, _ = cv2.Rodrigues(rvec_ref)

                        # 최종 상대 좌표 계산
                        relative_pos = np.dot(R_ref.T, final_cube_center_cam - tvec_ref)
                        
                        x, y, z = relative_pos.flatten()
                        text = f"Cube Center @ Ref 0: ({x:.3f}, {y:.3f}, {z:.3f}) m"
                        
                        # 결과를 화면에 표시하기 위해 2D로 투영
                        img_points, _ = cv2.projectPoints(final_cube_center_cam, np.zeros(3), np.zeros(3), cmtx, dist)
                        pt_2d = (int(img_points[0][0][0]), int(img_points[0][0][1]))
                        
                        cv2.circle(frame, pt_2d, 8, (0, 255, 255), -1)
                        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Cube Tracker", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()