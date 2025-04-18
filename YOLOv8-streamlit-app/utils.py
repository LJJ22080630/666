#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description:   Modified to support direct detection from processed image
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from io import BytesIO
import os


def _display_detected_frames(conf, model, st_frame, image):
    """
    显示YOLOv8检测结果到视频帧
    """
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Image',
                   channels="BGR",
                   use_container_width=True  # 已修复
                   )
    return res


@st.cache_resource
def load_model(model_path):
    """加载 YOLOv8 模型"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None


def infer_uploaded_image(conf, model, custom_image=None, patient_id=None):
    """
    单张图片检测流程
    :param conf: YOLOv8模型的置信度阈值
    :param model: YOLO模型对象
    :param custom_image: 图像路径或 BytesIO 对象
    :param patient_id: （可选）患者ID，用于在界面上标识结果
    :return: 检测结果列表（包含 eye、class_name、confidence）
    """
    col1, col2 = st.columns(2)
    results_list = []

    if custom_image is not None:
        # 加载图像
        if isinstance(custom_image, str):
            if os.path.exists(custom_image):
                image = Image.open(custom_image)
                st.session_state.detection_image = image
            else:
                st.error("处理后的图片未找到！")
                return []
        else:
            image = Image.open(custom_image)
            st.session_state.detection_image = image

        with col1:
            st.image(image, caption="处理后的图片", use_container_width=True)

        with st.spinner("正在进行检测..."):
            res = model.predict(image, conf=conf)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]

            with col1:
                st.session_state.result_image = res_plotted
                st.image(res_plotted, caption="检测结果图", use_container_width=True)

            # 类别映射
            disease_map = {
                "n": "正常眼底",
                "a": "AMD",
                "d": "糖尿病",
                "h": "高血压",
                "m": "近视",
                "g": "青光眼",
                "c": "白内障",
                "o": "其他疾病"
            }

            with col2:
                # 标题：检测结果（带上患者ID）
                if patient_id:
                    st.markdown(f"### 患者 {patient_id} 的检测结果")
                else:
                    st.markdown("### 检测结果")

                if len(boxes) == 0:
                    st.info("未检测到目标。")
                else:
                    image_width = image.width
                    box_info = []

                    for box in boxes:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_code = model.names[cls_id]
                        disease_name = disease_map.get(class_code, f"未知类别 ({class_code})")

                        x_center = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                        eye = "左眼" if x_center < image_width / 2 else "右眼"

                        box_info.append({
                            "class_name": disease_name,
                            "confidence": confidence,
                            "eye": eye
                        })

                    # 显示每个检测框结果
                    for info in box_info:
                        st.markdown(f"""
                        <div style='background-color: #f0f2f6; padding: 0.5rem; margin-bottom: 0.5rem; border-radius: 0.5rem;'>
                            <strong>{info["eye"]}预测疾病：</strong> {info["class_name"]}<br>
                            <strong>概率：</strong> {info["confidence"]:.2f}
                        </div>
                        """, unsafe_allow_html=True)
                        results_list.append(info)

                    # 标题：最终预测结论（带上患者ID）
                    st.markdown("---")
                    if patient_id:
                        st.markdown(f"### 患者 {patient_id} 的最终预测结论")
                    else:
                        st.markdown("### 最终预测结论")

                    normal_label = "正常眼底"
                    abnormal_diseases = [r["class_name"] for r in results_list if r["class_name"] != normal_label]

                    if not abnormal_diseases:
                        st.success("该患者眼底检查结果为正常。")
                    else:
                        disease_summary = "、".join(set(abnormal_diseases))
                        st.error(f"该患者可能患有以下疾病：{disease_summary}")

        return results_list
    else:
        return []


def infer_uploaded_video(conf, model):
    """
    视频检测流程
    """
    source_video = st.sidebar.file_uploader(
        label="请选择视频文件..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("开始检测"):
            with st.spinner("检测中..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(tfile.name)
                    st_frame = st.empty()
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf, model, st_frame, image)
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"视频处理错误: {e}")


def infer_uploaded_webcam(conf, model):
    """
    摄像头检测流程
    """
    try:
        flag = st.button("停止运行")
        vid_cap = cv2.VideoCapture(0)
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf, model, st_frame, image)
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"摄像头加载错误: {str(e)}")