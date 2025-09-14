import cv2
from pydantic import BaseModel
from typing import Dict, Any, Tuple
import mediapipe as mp
from deepface import DeepFace
import math
import time
import numpy as np
from fastapi import FastAPI
from mediapipe.python.solutions import pose

# ==================== 可调参数 ====================
CAM_INDEX = 0 # 摄像头序号
FRAME_W, FRAME_H = 640,480
FACE_FOCUSED_YAW_DEG = 20 # 左右偏头阙值
FACE_FOCUSED_PITCH_DEG = 15 # 上下点头阙值
EAR_CLOSED_THRESH = 0.20 # 眼睛闭合阙值（越小越闭）
NO_FACE_UNFOCUSED_GRACE = 0.6 # 无人脸但有人体时，判断不专注的权重

# cap = None  # 不在 import 时就打开，避免 --reload 多进程踩资源
app = FastAPI(title="Free Emotion & Attention API",version="1.0")

# 复用MediaPipe组件
mp_face_mesh = mp.solutions.face_mesh
# mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces= 1,
    refine_landmarks  = True,
    min_detection_confidence= 0.5,
    min_tracking_confidence= 0.5
)

pose = pose.Pose(
    static_image_mode=False,
    model_complexity=0,   # ★ 轻量模型，只用 upper-body 关键点
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 复用摄像头
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_H)

# 预热DeepFace模型(首次会自动下载)
# _EMOTION_MODEL = DeepFace.build_model("Emotion")

class EmotionResult(BaseModel):
    dominant_emotion: str
    emotions: Dict[str,float]
    frame_ts: float

class AttentionResult(BaseModel):
    focused: bool
    score: float # 0~1
    reason: str
    yaw_deg: float = None
    pitch_deg: float = None
    eye_open_prob: float = None # 简易眼睛张开概率（来自 EAR 映射）
    frame_ts: float

# --------- 工具函数 ---------
def grab_frame()-> np.ndarray:
    ok,frame = cap.read()
    if not ok:
        raise RuntimeError("无法从摄像头")
    return cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)

def deepface_emotion(frame_rgb: np.ndarray)->Tuple[str,Dict[str,float]]:
    # DeepFace 返回每种情绪的概率（百分比），转成 0~1
    analysis = DeepFace.analyze(
        img_path = frame_rgb,
        actions = ['emotion'],
        enforce_detection= False,
        detector_backend= 'opencv',
        # prog_bar = False
    )
    if isinstance(analysis,list) and analysis:
        analysis = analysis[0]
    emos = analysis.get('emotion',{})
    if not emos:
        return "neutral",{"neutral":1.0}
    total = sum(emos.values()) or 1.0
    emos01 = {k.lower():float(v)/total for k,v in emos.items()}
    dominant = analysis.get('dominant_emotion',max(emos01,key = emos01.get))
    return dominant.lower(),emos01

# 眼睛 EAR（Eye Aspect Ratio）
# 使用 FaceMesh 索引：左眼(33, 160, 158, 133, 153, 144)，右眼(263, 387, 385, 362, 380, 373)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def ear_of_eye(pts):
    # pts: 6 x (x,y)
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    p1, p2, p3, p4, p5, p6 = [np.array(p, dtype=np.float32) for p in pts]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    if h == 0:
        return 0.0
    return float((v1 + v2) / (2.0 * h))

# 基于 6 个 3D-2D 对应点的简易头部姿态解算（solvePnP）
# 取鼻梁、眼角、嘴角等近似 3D 模型点（单位：毫米，比例只影响尺度不影响角度）
MODEL_3D = np.array([
    [0.0,   0.0,   0.0],   # 鼻尖（NOSE）
    [-30.0, -30.0, -30.0], # 左眼角近似
    [ 30.0, -30.0, -30.0], # 右眼角近似
    [-25.0,  30.0, -30.0], # 左嘴角近似
    [ 25.0,  30.0, -30.0], # 右嘴角近似
    [0.0,   -65.0, -20.0], # 下巴近似
], dtype=np.float32)

# FaceMesh 对应的 2D 点索引（鼻尖, 左眼外角, 右眼外角, 左嘴角, 右嘴角, 下巴）
MESH_IDX = [1, 33, 263, 61, 291, 199]

def estimate_head_pose(landmarks, img_w, img_h):
    pts_2d = []
    for idx in MESH_IDX:
        lm = landmarks[idx]
        pts_2d.append([lm.x * img_w, lm.y * img_h])
    pts_2d = np.array(pts_2d, dtype=np.float32)

    focal = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([[focal, 0, center[0]],
                              [0, focal, center[1]],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeff = np.zeros((4, 1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, pts_2d, camera_matrix, dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None

    rot_mat, _ = cv2.Rodrigues(rvec)
    # 由旋转矩阵取欧拉角（RzRyRx），这里简化成 yaw/pitch（度）
    sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    pitch = math.degrees(math.atan2(-rot_mat[2, 0], sy))  # x 轴
    yaw = math.degrees(math.atan2(rot_mat[1, 0], rot_mat[0, 0]))  # y 轴
    return yaw, pitch

def attention_from_facemesh_results(image_rgb) -> Dict[str, Any]:
    h, w = image_rgb.shape[:2]
    res = face_mesh.process(image_rgb)
    if not res.multi_face_landmarks:
        return {"has_face": False}

    lms = res.multi_face_landmarks[0].landmark

    # 计算左右眼 EAR
    def get_pts(indices):
        return [(lms[i].x * w, lms[i].y * h) for i in indices]
    ear_left = ear_of_eye(get_pts(LEFT_EYE))
    ear_right = ear_of_eye(get_pts(RIGHT_EYE))
    ear = (ear_left + ear_right) / 2.0

    # 粗略头部姿态
    yaw, pitch = estimate_head_pose(lms, w, h)
    eyes_open_prob = float(max(0.0, min(1.0, (ear - 0.12) / (0.32 - 0.12))))  # 把 EAR ~[0.12,0.32] 映射到 [0,1]

    return {
        "has_face": True,
        "yaw_deg": float(yaw) if yaw is not None else None,
        "pitch_deg": float(pitch) if pitch is not None else None,
        "ear": float(ear),
        "eyes_open_prob": eyes_open_prob,
    }

def body_presence_and_orientation(image_rgb) -> Dict[str, Any]:
    h, w = image_rgb.shape[:2]
    res = pose.process(image_rgb)
    if not res.pose_landmarks:
        return {"has_body": False}

    # 用肩膀关键点粗略估计身体朝向：左肩 11，右肩 12
    lm = res.pose_landmarks.landmark
    left_sh = lm[11]
    right_sh = lm[12]
    if (left_sh.visibility < 0.5) and (right_sh.visibility < 0.5):
        return {"has_body": False}

    # 肩线与水平的夹角（度），肩线太斜可能表示显著侧身
    dx = (right_sh.x - left_sh.x) * w
    dy = (right_sh.y - left_sh.y) * h
    shoulder_angle_deg = abs(math.degrees(math.atan2(dy, dx)))
    return {"has_body": True, "shoulder_angle_deg": float(shoulder_angle_deg)}

def compute_attention_score(face_info: Dict[str, Any], body_info: Dict[str, Any]) -> Tuple[float, str, bool, float, float, float]:
    """
    返回：score(0~1), reason, focused(bool), yaw, pitch, eyes_open_prob
    """
    yaw = pitch = eyes_open_prob = None

    if face_info.get("has_face"):
        yaw = face_info.get("yaw_deg")
        pitch = face_info.get("pitch_deg")
        eyes_open_prob = face_info.get("eyes_open_prob", 0.5)

        # 视线/姿态分
        pose_score = 1.0
        if yaw is not None:
            pose_score *= max(0.0, 1.0 - abs(yaw) / FACE_FOCUSED_YAW_DEG)
        if pitch is not None:
            pose_score *= max(0.0, 1.0 - abs(pitch) / FACE_FOCUSED_PITCH_DEG)
        pose_score = max(0.0, min(1.0, pose_score))

        # 眼睛分（闭眼降低专注）
        eye_score = eyes_open_prob  # 已是 0~1

        # 综合（简单乘积 + 温和放缩）
        score = (0.6 * pose_score + 0.4 * eye_score)
        score = max(0.0, min(1.0, score))

        focused = (score >= 0.5)
        reason = []
        if yaw is not None and abs(yaw) > FACE_FOCUSED_YAW_DEG:
            reason.append(f"头部偏转大(|yaw|>{FACE_FOCUSED_YAW_DEG}°)")
        if pitch is not None and abs(pitch) > FACE_FOCUSED_PITCH_DEG:
            reason.append(f"点头/仰头大(|pitch|>{FACE_FOCUSED_PITCH_DEG}°)")
        if eyes_open_prob < 0.3:
            reason.append("疑似闭眼/频繁眨眼")
        if not reason:
            reason.append("面向屏幕且双眼张开")

        return score, "；".join(reason), focused, yaw, pitch, eyes_open_prob

    # 没有人脸：看身体
    if body_info.get("has_body"):
        # 肩线夹角大 + 没有人脸 => 大概率未面向屏幕
        shoulder_angle = body_info.get("shoulder_angle_deg", 0.0)
        # 无人脸时给一个较低分
        base = 1.0 - NO_FACE_UNFOCUSED_GRACE
        # 肩线越“斜”，越不专注
        shoulder_penalty = min(1.0, shoulder_angle / 45.0)  # 45° 以上基本侧身
        score = max(0.0, base * (1.0 - 0.7 * shoulder_penalty))
        focused = score >= 0.5
        reason = "未检出人脸，但检测到人体；根据身体朝向推断"
        return score, reason, focused, yaw, pitch, eyes_open_prob

    # 连人体都没有
    return 0.0, "画面中无人/遮挡严重", False, yaw, pitch, eyes_open_prob


# ------------------- API 路由 -------------------
@app.get("/analyze_emotion", response_model=EmotionResult)
def analyze_emotion():
    frame = grab_frame()
    dominant, emotions01 = deepface_emotion(frame)
    return EmotionResult(
        dominant_emotion=dominant,
        emotions=emotions01,
        frame_ts=time.time()
    )

@app.get("/attention", response_model=AttentionResult)
def attention():
    frame = grab_frame()
    face_info = attention_from_facemesh_results(frame)
    body_info = {}   # 先不依赖 pose

    score, reason, focused, yaw, pitch, eyes_open_prob = compute_attention_score(face_info, body_info)
    return AttentionResult(
        focused=bool(focused),
        score=float(score),
        reason=reason,
        yaw_deg=yaw,
        pitch_deg=pitch,
        eyes_open_prob=eyes_open_prob,
        frame_ts=time.time()
    )
