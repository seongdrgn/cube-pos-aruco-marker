import cv2
import numpy as np

def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)           # (3,3)
    t = tvec.reshape(3, 1)               # (3,1)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = t[:, 0]
    return T

def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3:4]
    Tinv = np.eye(4, dtype=float)
    Tinv[:3, :3] = R.T
    Tinv[:3,  3] = (-R.T @ t)[:, 0]
    return Tinv

def T_to_rvec_tvec(T):
    R = T[:3, :3]
    t = T[:3,  3]
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3), t.reshape(3)

def relative_poses_from_id0(ids, rvecs, tvecs):
    """0번 마커 기준으로 각 마커의 상대 포즈 반환
    반환: dict[marker_id] = (rvec_rel, tvec_rel)  # 모두 0번 좌표계 기준
    """
    ids = ids.flatten()
    # 0번이 반드시 검출되어야 합니다.
    if 0 not in ids:
        raise ValueError("ID 0 marker not detected in this frame.")

    # 딕셔너리로 포즈 정리
    pose_dict = {}
    for i, mid in enumerate(ids):
        T_c_mi = rvec_tvec_to_T(rvecs[i,0], tvecs[i,0])
        pose_dict[int(mid)] = T_c_mi

    T_c_m0 = pose_dict[0]           # 카메라←0번
    T_m0_c = invert_T(T_c_m0)       # 0번←카메라

    rel = {}
    for mid, T_c_mi in pose_dict.items():
        T_m0_mi = T_m0_c @ T_c_mi   # 0번←i번
        rvec_rel, tvec_rel = T_to_rvec_tvec(T_m0_mi)
        rel[mid] = (rvec_rel, tvec_rel)

    return rel