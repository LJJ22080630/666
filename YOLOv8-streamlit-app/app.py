from pathlib import Path
from PIL import Image
import streamlit as st
import subprocess
import os
import tempfile
import cv2
import numpy as np
from io import BytesIO
import time
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam
from pathlib import Path
from utils import load_model  # 确保有这个导入
from pathlib import Path
import tempfile

# 基础路径
BASE_DIR = Path(__file__).resolve().parent

# 获取资源路径
def get_asset_path(filename):
    return BASE_DIR / "assets" / filename

def get_model_path(filename):
    return BASE_DIR / "models" / filename

# 用于临时文件保存
def get_temp_file(filename):
    return Path(tempfile.gettempdir()) / filename
# 登录账户信息
USER_CREDENTIALS = {
    "admin": "123456",
    "doctor": "abc123"
}

# 初始化session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.5  # 默认置信度
if 'result_image' not in st.session_state:
    st.session_state.result_image = None
if 'detection_triggered' not in st.session_state:
    st.session_state.detection_triggered = False
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'merged_image_path' not in st.session_state:
    st.session_state.merged_image_path = None
if 'grey_image_path' not in st.session_state:
    st.session_state.grey_image_path = None
if 'test_image_path' not in st.session_state:
    st.session_state.test_image_path = None
if 'processing_steps' not in st.session_state:
    st.session_state.processing_steps = {
        'merge_complete': False,
        'grey_complete': False,
        'test_complete': False
    }
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'selected_folders' not in st.session_state:
    st.session_state.selected_folders = []

# 设置页面布局
st.set_page_config(
    page_title="瞳芯智鉴平台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加自定义CSS样式
st.markdown("""
<style>
.title-wrapper {
    background-color: #1e3a8a;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 2rem;
}
.title-wrapper h1 {
    color: white;
    margin: 0;
}
.custom-button {
    background-color: #1e3a8a;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    border: none;
    font-size: 1rem;
    cursor: pointer;
    margin-top: 1rem;
    margin-right: 1rem;
}
.custom-button:hover {
    background-color: #1a3377;
}
.image-container {
    display: flex;
    justify-content: space-around;
    margin-top: 2rem;
}
.image-box {
    text-align: center;
    margin: 0 1rem;
}
.processing-steps {
    margin-top: 2rem;
    margin-bottom: 2rem;
}
.detection-results {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-top: 1rem;
}
.result-item {
    margin: 0.5rem 0;
    padding: 0.5rem;
    background-color: white;
    border-radius: 0.25rem;
}
.folder-selector {
    border: 2px dashed #ccc;
    padding: 20px;
    border-radius: 5px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# 使用自定义样式包装标题
st.markdown("""
<div class="title-wrapper">
    <h1>瞳芯智鉴--基于YOLOv8多模态特征融合的眼底病灶诊断系统</h1>
</div>
""", unsafe_allow_html=True)


def show_history_page():
    st.markdown("""
    <style>
        .history-header {
            background-color: #1e3a8a;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
        }
        .history-content {
            font-family: 'Courier New', monospace;
            white-space: pre;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
            max-height: 60vh;
            overflow-y: auto;
            line-height: 1.5;
        }
        .history-actions {
            margin-top: 1rem;
            display: flex;
            gap: 1rem;
        }
    </style>

    <div class="history-header">
        <h1>历史检测记录</h1>
    </div>
    """, unsafe_allow_html=True)

    # 获取返回目标页面，默认为home
    return_to = st.session_state.get('return_to', 'test')

    # 操作按钮行
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        if st.button(f"← 返回{'测试页面' if return_to == 'test' else '主页面'}"):
            st.session_state.page = return_to
            st.rerun()
    with col2:
        if st.button("🔄 刷新记录"):
            st.rerun()

    # ... 历史记录页面其余内容保持不变 ...

    # 记录内容区域
    if os.path.exists("检测结果.txt"):
        with open("检测结果.txt", "r", encoding="utf-8") as f:
            content = f.read().strip()

        if content:
            # 显示记录内容
            st.markdown(f'<div class="history-content">{content}</div>',
                        unsafe_allow_html=True)

            # 操作按钮
            st.markdown("---")
            st.markdown("### 记录操作")

            col1, col2 = st.columns(2)
            with col1:
                # 下载按钮
                st.download_button(
                    "💾 下载完整记录",
                    data=content,
                    file_name=f"眼底检测记录_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                # 清空记录按钮
                if st.button("🗑️ 清空历史记录", use_container_width=True):
                    try:
                        os.remove("检测结果.txt")
                        st.success("历史记录已清空")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"清空记录失败: {e}")
        else:
            st.warning("历史记录文件为空")
    else:
        st.warning("尚未生成任何检测记录")

    # 添加空白区域使页面更平衡
    st.markdown("<div style='margin-top: 3rem;'></div>",
                unsafe_allow_html=True)
def save_detection_results(patient_id, results, filename="检测结果.txt"):
    """将检测结果保存到文本文件"""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n\n=== 患者 {patient_id} 检测结果 ===\n")
        f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        if not results:
            f.write("未检测到任何疾病\n")
            return

        # 记录各项检测结果
        for result in results:
            eye = result.get("eye", "未知眼别")
            f.write(f"{eye}: {result['class_name']} (置信度: {result['confidence']:.2f})\n")

        # 记录最终结论
        normal_label = "正常眼底"
        abnormal_diseases = [r["class_name"] for r in results if r["class_name"] != normal_label]

        if not abnormal_diseases:
            f.write("最终结论: 眼底检查结果正常\n")
        else:
            disease_summary = "、".join(set(abnormal_diseases))
            f.write(f"最终结论: 可能患有{disease_summary}\n")


# 登录表单
def login_modal():
    with st.form("login_form", clear_on_submit=True):
        st.subheader("请输入账号密码")
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        login_submit = st.form_submit_button("登录")

        if login_submit:
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.show_login = False
                st.session_state.page = "test"
                st.success("登录成功，正在跳转测试页...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("账号或密码错误，请重试。")


# 重置处理状态
def reset_processing_states():
    st.session_state.uploaded_images = []
    st.session_state.merged_image_path = None
    st.session_state.grey_image_path = None
    st.session_state.test_image_path = None
    st.session_state.processing_steps = {
        'merge_complete': False,
        'grey_complete': False,
        'test_complete': False
    }
    st.session_state.detection_triggered = False
    st.session_state.detection_results = []
    st.session_state.selected_folders = []
    st.session_state.result_image = None  # 关键新增，清除检测结果图像


def merge_images(image_paths):
    try:
        output_path = "merged_result.jpg"
        cmd = ["python", "add.py", image_paths[0], image_paths[1], output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            st.session_state.merged_image_path = output_path
            st.session_state.processing_steps['merge_complete'] = True
            return True
        else:
            st.error(f"拼接失败: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"发生错误: {str(e)}")
        return False


def run_grey_processing():
    try:
        input_path = "merged_result.jpg"
        output_path = "grey_result.jpg"

        if not os.path.exists(input_path):
            st.error(f"拼接后的图片不存在: {input_path}")
            return False

        cmd = ["python", "grey.py", input_path, output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            if os.path.exists(output_path):
                st.session_state.grey_image_path = output_path
                st.session_state.processing_steps['grey_complete'] = True
                return True
            else:
                st.error("灰度化成功但输出文件未生成")
                return False
        else:
            st.error(f"灰度化处理失败: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"发生错误: {str(e)}")
        return False


def run_test_processing():
    try:
        input_path = "grey_result.jpg"
        output_path = "test_result.jpg"

        if not os.path.exists(input_path):
            st.error(f"灰度化图片不存在: {input_path}")
            return None

        cmd = ["python", "test.py", input_path, output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            if os.path.exists(output_path):
                st.session_state.test_image_path = output_path
                st.session_state.processing_steps['test_complete'] = True
                return output_path
        return None
    except Exception as e:
        st.error(f"发生错误: {str(e)}")
        return None


def display_detection_results(results):
    if not results:
        st.warning("没有检测到任何疾病。")
        return

    st.markdown("### 检测结果")
    st.markdown("<div class='detection-results'>", unsafe_allow_html=True)

    for result in results:
        eye = result.get("eye", "未知眼别")
        st.markdown(f"""
        <div class='result-item'>
            <strong>{eye}预测疾病：</strong> {result['class_name']}<br>
            <strong>概率：</strong> {result['confidence']:.2f}
        </div>
        """, unsafe_allow_html=True)

    # 最终结论
    st.markdown("### 最终预测结论")
    normal_label = "正常眼底"
    abnormal_diseases = [r["class_name"] for r in results if r["class_name"] != normal_label]

    if not abnormal_diseases:
        st.success("该患者眼底检查结果为正常。")
    else:
        disease_summary = "、".join(set(abnormal_diseases))
        st.error(f"该患者可能患有以下疾病：{disease_summary}")

    st.markdown("</div>", unsafe_allow_html=True)


def process_patient_images(image_paths, patient_id):
    """处理单个患者的图片"""
    with st.spinner(f"正在处理患者 {patient_id} 的图片..."):
        try:
            # 1. 拼接图片
            merged_path = f"temp_{patient_id}_merged.jpg"
            cmd = ["python", "add.py", image_paths[0], image_paths[1], merged_path]
            merge_result = subprocess.run(cmd, capture_output=True, text=True)

            if merge_result.returncode != 0:
                st.error(f"拼接失败: {merge_result.stderr}")
                return False

            # 显示拼接结果
            st.image(merged_path, caption=f"拼接结果 - 患者 {patient_id}", use_container_width=True)

            # 2. 灰度处理
            grey_path = f"temp_{patient_id}_grey.jpg"
            cmd = ["python", "grey.py", merged_path, grey_path]
            grey_result = subprocess.run(cmd, capture_output=True, text=True)

            if grey_result.returncode != 0:
                st.error(f"灰度化失败: {grey_result.stderr}")
                return False

            # 显示灰度结果
            st.image(grey_path, caption=f"灰度图像 - 患者 {patient_id}", use_container_width=True)

            # 3. 测试处理
            test_path = f"temp_{patient_id}_test.jpg"
            cmd = ["python", "test.py", grey_path, test_path]
            test_result = subprocess.run(cmd, capture_output=True, text=True)

            if test_result.returncode != 0:
                st.error(f"测试处理失败: {test_result.stderr}")
                return False

            # 显示测试处理结果
            st.image(test_path, caption=f"测试处理图像 - 患者 {patient_id}", use_container_width=True)

            # 4. 进行检测
            with open(test_path, "rb") as f:
                img_bytes = f.read()
            img_file = BytesIO(img_bytes)
            img_file.name = test_path

            results = infer_uploaded_image(
                st.session_state.confidence,
                model,
                custom_image=img_file,
                patient_id=patient_id  # 新增这一行

            )

            if results:
                display_detection_results(results)
            else:
                st.warning(f"患者 {patient_id} 图像中未检测到疾病")

            return True

        except Exception as e:
            st.error(f"处理患者 {patient_id} 出错: {e}")
            return False
        finally:
            # 清理临时文件
            for temp_file in [merged_path, grey_path, test_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

# 模型加载
default_model_path = get_model_path("best.pt")
model = None
if default_model_path.exists():
    try:
        model = load_model(default_model_path)
    except Exception as e:
        st.error(f"模型加载失败：{e}")
else:
    st.error("默认模型 best.pt 未找到")

# 首页内容
if st.session_state.page == 'home':
    image_path = get_asset_path('微信图片_20250402140412.png')
    st.image(str(image_path), caption='瞳芯智鉴--基于YOLOv8多模态特征融合的眼底病灶诊断系统', use_container_width=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("在线测试", key="home_online_test_button"):
            st.session_state.show_login = True
    with col2:
        if st.button("历史记录", key="history_button"):
            st.session_state.page = "history"
            st.rerun()
    # 首页底部图文并排展示（图片大+文字）
    st.markdown("### 本项目聚焦7类常见眼疾，通过深度学习技术实现高精度自动诊断，助力基层医疗与远程筛查。")

    # 第一组：图文并排（放大图片）
    col1, col2 = st.columns([3, 2])  # 图片列更宽
    with col1:
        st.image(str(get_asset_path("微信图片_20250413163617.png")), use_container_width=True)
    with col2:
        st.markdown("""
        #### 系统特点
        - 集成改进型 **YOLOv8**
        - 支持 **七类常见眼疾** 同步识别
        - 分类标注各类病变区域
        - 输出 **可疑区域 + 置信度值**
        - 适用于基层医院和远程筛查
        """)

    st.markdown("---")

    # 第二组：图文并排（放大图片）
    col3, col4 = st.columns([3, 2])  # 图片列更宽
    with col3:
        st.image(str(get_asset_path("微信图片_20250413163640.png")), use_container_width=True)
    with col4:
        st.markdown("""
        #### 七种支持识别的疾病：
                                     
                "n": "正常眼底",
                "a": "AMD",
                "d": "糖尿病",
                "h": "高血压",
                "m": "近视",
                "g": "青光眼",
                "c": "白内障",
                "o": "其他疾病"

        > 检测准确率和召回率 **均超90%**，增强结果可信度。
        """)

    if st.session_state.show_login and not st.session_state.logged_in:
        login_modal()

    st.markdown("<div style='margin-top: 5rem;'></div>", unsafe_allow_html=True)
elif st.session_state.page == 'history':
    # 历史记录页面
    show_history_page()
# 在测试页面内容部分 (elif st.session_state.page == 'test') 修改以下内容：

elif st.session_state.page == 'test':
    if not st.session_state.logged_in:
        st.warning("请先登录后再进入测试页面。")
        st.session_state.page = 'home'
        st.rerun()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← 返回首页", key="test_to_home"):
            st.session_state.page = 'home'
            st.rerun()
    with col2:
        if st.button("📜 查看历史记录", key="test_to_history"):
            st.session_state.page = 'history'
            st.session_state.return_to = 'test'  # 设置返回目标
            st.rerun()
    # 模式选择
    st.markdown("### 检测模式选择")
    detection_mode = st.radio(
        "请选择检测模式：",
        ("单用户检测（拼接+预处理）", "批量识别（多图独立检测）"),
        key="mode_selector"
    )
    st.markdown("## 测试页面")

    if st.button("重新上传图片", key="reset_upload_button"):
        reset_processing_states()
        st.rerun()

    if detection_mode == "批量识别（多图独立检测）":
        st.markdown("""
           <div class="folder-selector">
               <h4>文件夹选择说明</h4>
               <p>请选择包含患者图片的文件夹，系统会自动处理所有患者的图片</p>
               <p>每个患者应有2张图片，命名格式：<code>患者ID_左眼.jpg</code> 和 <code>患者ID_右眼.jpg</code></p>
           </div>
           """, unsafe_allow_html=True)

        # 使用文件上传器模拟文件夹选择
        uploaded_files = st.file_uploader(
            "选择包含患者图片的文件夹（全选文件夹中的图片）",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="folder_uploader"
        )

        if uploaded_files:
            # 按文件夹分组
            folder_dict = {}
            for file in uploaded_files:
                folder_name = os.path.dirname(file.name)
                if folder_name not in folder_dict:
                    folder_dict[folder_name] = []
                folder_dict[folder_name].append(file)

            # 处理所有文件夹
            if st.button("批量处理所有患者", key="process_all_patients"):
                all_results = {}
                progress_bar = st.progress(0)
                total_patients = sum(len(patient_dict) for patient_dict in folder_dict.values())
                processed = 0

                for folder_name, files in folder_dict.items():
                    # 按患者ID分组
                    patient_dict = {}
                    for file in files:
                        file_name = file.name.lower()
                        if "_" in file_name:
                            patient_id = file_name.split("_")[0]
                        else:
                            patient_id = os.path.splitext(file_name)[0]

                        if patient_id not in patient_dict:
                            patient_dict[patient_id] = []
                        patient_dict[patient_id].append(file)

                    # 处理每个患者
                    for patient_id, patient_files in patient_dict.items():
                        if len(patient_files) != 2:
                            st.warning(f"患者 {patient_id} 的图片数量不是2张，跳过处理")
                            continue

                        try:
                            # 更新进度
                            processed += 1
                            progress_bar.progress(processed / total_patients)

                            # 保存临时文件
                            temp_files = []
                            for i, img_file in enumerate(patient_files):
                                temp_path = f"temp_{patient_id}_{i}.jpg"
                                with open(temp_path, "wb") as f:
                                    img_file.seek(0)  # 确保文件指针在开头
                                    f.write(img_file.getbuffer())
                                temp_files.append(temp_path)

                            # 处理流程
                            merged_path = get_temp_file(f"temp_{patient_id}_merged.jpg")
                            grey_path = get_temp_file(f"temp_{patient_id}_grey.jpg")
                            test_path = get_temp_file(f"temp_{patient_id}_test.jpg")

                            # 1. 拼接图片
                            cmd = ["python", "add.py", temp_files[0], temp_files[1], merged_path]
                            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                            # 2. 灰度处理
                            cmd = ["python", "grey.py", merged_path, grey_path]
                            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                            # 3. 测试处理
                            cmd = ["python", "test.py", grey_path, test_path]
                            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                            # 4. 进行检测
                            with open(test_path, "rb") as f:
                                img_bytes = f.read()
                            img_file = BytesIO(img_bytes)
                            img_file.name = test_path

                            results = infer_uploaded_image(
                                st.session_state.confidence,
                                model,
                                custom_image=img_file,
                                patient_id=patient_id  # 新增这一行
                            )


                            if results:
                                all_results[patient_id] = results
                                save_detection_results(patient_id, results)
                            else:
                                all_results[patient_id] = None
                                save_detection_results(patient_id, [])

                        except subprocess.CalledProcessError as e:
                            st.error(f"患者 {patient_id} 处理失败: {e.stderr.decode('utf-8')}")
                            all_results[patient_id] = None
                        except Exception as e:
                            st.error(f"患者 {patient_id} 发生错误: {str(e)}")
                            all_results[patient_id] = None
                        finally:
                            # 清理临时文件
                            for temp_file in temp_files + [merged_path, grey_path, test_path]:
                                if os.path.exists(temp_file):
                                    try:
                                        os.remove(temp_file)
                                    except:
                                        pass
                        # 批量处理完成后，保存汇总结果
                with open("检测结果.txt", "a", encoding="utf-8") as f:
                    f.write("\n\n=== 批量检测汇总 ===")
                    f.write(f"\n检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    f.write(f"\n总患者数: {len(all_results)}")
                    f.write(f"\n完成检测: {sum(1 for r in all_results.values() if r is not None)}")
                    f.write(f"\n失败检测: {sum(1 for r in all_results.values() if r is None)}")

                # 显示所有结果
                st.success("所有患者处理完成！")
                progress_bar.empty()

                # 显示汇总结果
                st.markdown("## 批量处理结果汇总")
                for patient_id, results in all_results.items():
                    with st.expander(f"患者 {patient_id} 的检测结果"):
                        if results:
                            display_detection_results(results)
                        else:
                            st.warning("未能完成检测")

    elif detection_mode == "单用户检测（拼接+预处理）":
        # 保持原有的单用户检测逻辑不变
        uploaded_files = st.file_uploader(
            "请上传两张图片（左右眼）",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="single_file_uploader"
        )

        if uploaded_files:
            if len(uploaded_files) > 2:
                st.warning("只能上传两张图片用于拼接")
                uploaded_files = uploaded_files[:2]
            st.session_state.uploaded_images = uploaded_files[:2]


    if st.session_state.uploaded_images:
        st.markdown("### 已上传的图片")
        cols = st.columns(2)
        for i, img_file in enumerate(st.session_state.uploaded_images):
            with cols[i % 2]:
                st.image(img_file, caption=f"图片 {i + 1}", use_container_width=True)

        st.markdown("### 处理步骤")

        if len(st.session_state.uploaded_images) == 2 and st.button("拼接左右眼", key="test_merge_button"):
            filenames = [img.name.lower() for img in st.session_state.uploaded_images]
            try:
                ids = [f.split("_")[0] for f in filenames if "_" in f]
                if len(ids) == 2 and ids[0] != ids[1]:
                    st.warning("警告：两张图像可能不属于同一位患者")
                else:
                    with st.spinner("正在拼接图片..."):
                        temp_files = []
                        try:
                            for i, img_file in enumerate(st.session_state.uploaded_images):
                                temp_file = f"temp_{i}.jpg"
                                with open(temp_file, "wb") as f:
                                    f.write(img_file.getbuffer())
                                temp_files.append(temp_file)

                            if merge_images(temp_files):
                                st.success("图片拼接成功!")
                                st.rerun()
                        finally:
                            for temp_file in temp_files:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
            except Exception as e:
                st.error(f"文件名解析错误：{e}")

        if st.session_state.processing_steps['merge_complete'] and os.path.exists("merged_result.jpg"):
            st.image("merged_result.jpg", caption="拼接后的图片", use_container_width=True)

            if st.button("预处理1：灰度化", key="test_grey_button"):
                with st.spinner("正在灰度化处理..."):
                    if run_grey_processing():
                        st.success("灰度化处理成功!")
                        st.rerun()

            if st.session_state.processing_steps['grey_complete'] and os.path.exists("grey_result.jpg"):
                st.image("grey_result.jpg", caption="灰度化后的图片", use_container_width=True)

                if st.button("预处理2：测试处理", key="test_process_button"):
                    with st.spinner("正在测试处理..."):
                        if run_test_processing():
                            st.success("测试处理成功!")
                            st.rerun()

                if st.session_state.processing_steps['test_complete'] and os.path.exists("test_result.jpg"):
                    st.image("test_result.jpg", caption="测试处理后的图片", use_container_width=True)

                    if st.button("进行检测", key="run_detection_button"):
                        st.session_state.detection_triggered = True
                        st.rerun()

if (
    st.session_state.page == 'test'
    and st.session_state.get("mode_selector") == "单用户检测（拼接+预处理）"
):
    if st.session_state.result_image is not None:
        st.image(st.session_state.result_image, caption="检测结果图", use_container_width=True)

    if st.session_state.detection_results:
        save_detection_results("单用户检测", st.session_state.detection_results)
        st.success("检测结果已保存到'检测结果.txt'")
        display_detection_results(st.session_state.detection_results)


# 侧边栏内容
st.sidebar.header("DL Model Config")
st.sidebar.markdown("---")
if st.sidebar.button("注销"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.show_login = False
    st.session_state.page = 'home'
    st.rerun()

# 模型配置
task_type = st.sidebar.selectbox(
    "选择任务类型",
    ["Detection"],
    key="sidebar_task_select"
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "选择检测模型",
        config.DETECTION_MODEL_LIST,
        key="sidebar_model_select"
    )
else:
    st.error("目前仅支持 Detection 模式")

# 置信度设置
st.session_state.confidence = float(st.sidebar.slider(
    "选择模型置信度", 30, 100, 50,
    key="confidence_slider"
)) / 100


# 输入源选择
source_selectbox = st.sidebar.selectbox(
    "选择输入源",
    config.SOURCES_LIST,
    key="sidebar_source_select"
)

# 检测触发逻辑
if st.session_state.get('detection_triggered', False):
    source_selectbox = config.SOURCES_LIST[0]
    st.session_state.detection_triggered = False

    if st.session_state.test_image_path and os.path.exists(st.session_state.test_image_path):
        with open(st.session_state.test_image_path, "rb") as f:
            img_bytes = f.read()

        img_file = BytesIO(img_bytes)
        img_file.name = "test_result.jpg"

        results = infer_uploaded_image(
            st.session_state.confidence,
            model,
            custom_image=img_file,
            patient_id="单用户检测"
        )

        if results:
            st.session_state.detection_results = results
            st.rerun()

# 其他输入源处理
elif source_selectbox == config.SOURCES_LIST[0]:  # Image
    results = infer_uploaded_image(st.session_state.confidence, model)
    if results:
        st.session_state.detection_results = results

elif source_selectbox == config.SOURCES_LIST[1]:  # Video
    results = infer_uploaded_video(st.session_state.confidence, model)
    if results:
        st.session_state.detection_results = results

elif source_selectbox == config.SOURCES_LIST[2]:  # Webcam
    results = infer_uploaded_webcam(st.session_state.confidence, model)
    if results:
        st.session_state.detection_results = results