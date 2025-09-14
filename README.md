# 🎯 Emotion & Attention Recognition API

本项目基于 **FastAPI + OpenCV + DeepFace + Mediapipe**，实现了实时摄像头输入的 **情绪识别** 与 **注意力识别**。  
提供 RESTful API 接口，开箱即用，无需额外训练模型。

---

这样一来，文档开头就有一个可视化的整体流程：  

- **摄像头输入**  
- → FastAPI 框架处理  
- → DeepFace 做情绪识别  
- → Mediapipe 做关键点检测（眼睛开合 + 头部姿态）  
- → 得到情绪结果与注意力结果  
- → 统一通过 API 返回 JSON  


---

## 📌 功能介绍

- **情绪识别**：调用 DeepFace 模型，返回人脸情绪分类及占比。  
- **注意力识别**：基于 Mediapipe FaceMesh 检测人脸关键点，结合头部姿态与眼睛开合度，计算注意力分数并判断是否专注。  

---

## 🚀 API 调用说明

### 1. 情绪识别接口
- **URL**: `/analyze_emotion`  
- **Method**: `GET`  
- **功能**: 捕获摄像头画面并返回人脸的情绪占比。  

### 2. 注意力识别接口
- **URL**: `/attention`  
- **Method**: `GET`  
- **功能**: 捕获摄像头画面并返回是否专注、专注分数、头部姿态、眼睛开合度等信息。  

---

## 🔍 注意力识别原理

1. **眼睛开合度 (EAR, Eye Aspect Ratio)**  
   - 使用 FaceMesh 关键点计算 EAR：  
     $$
     EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}
     $$  
   - EAR 越小表示闭眼，越大表示睁眼。通过线性映射到 0~1，得到 **eyes_open_prob**。

2. **头部姿态估计 (Yaw / Pitch)**  
   - 利用鼻尖、眼角、嘴角、下巴等关键点，结合 `cv2.solvePnP` 解算头部旋转角度。  
   - **Yaw**：头偏左右；**Pitch**：头低下或抬起。

3. **注意力分数计算**  
   - 视线/姿态分 (pose_score) + 眼睛分 (eye_score) 综合：  
     $$
     score = 0.6 \times pose\_score + 0.4 \times eye\_score
     $$  
   - 当 score ≥ 0.5，判定为 **专注状态**；否则为走神。

---

## ⚙️ 使用方法

### 1. 启动服务

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
服务启动后，访问 http://127.0.0.1:8000/docs
 查看交互式 API 文档。

### 2. 调用情绪识别接口
GET http://127.0.0.1:8000/analyze_emotion


返回示例：
```
{
  "dominant_emotion": "fear",
  "emotions": {
    "angry": 0.00001,
    "disgust": 0.0,
    "fear": 0.9989,
    "happy": 0.000002,
    "sad": 0.0008,
    "surprise": 0.000008,
    "neutral": 0.00028
  },
  "frame_ts": 1757854743.6871197
}
```
### 3. 调用注意力识别接口
GET http://127.0.0.1:8000/attention

```
返回示例：

{
  "focused": true,
  "score": 0.72,
  "reason": "面向屏幕且双眼张开",
  "yaw_deg": -5.3,
  "pitch_deg": 3.2,
  "eye_open_prob": 0.84,
  "frame_ts": 1757854852.123456
}
```

字段说明：

- focused: 是否专注 (true=专注, false=走神)
- score: 专注度分数 (0~1)
- reason: 判定原因（偏头/低头/闭眼/面向屏幕） 
- yaw_deg / pitch_deg: 头部姿态角度 
- eye_open_prob: 眼睛睁开概率 
- frame_ts: 时间戳

## 📎 注意事项

- 默认使用本机摄像头 (CAM_INDEX = 0)。如有多路摄像头可调整。 
- DeepFace 会在首次调用时自动下载预训练模型。 
- Mediapipe 可能需要 CPU 支持 AVX 指令集以加速。 
- 不要使用 uvicorn --reload，避免摄像头资源被多进程竞争
