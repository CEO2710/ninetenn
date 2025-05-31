import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 确保模型文件存在
if not os.path.exists('saved_models/preprocessor.pkl') or not os.path.exists('saved_models/best_model.pkl'):
    st.error("模型文件未找到！请确保已上传 preprocessor.pkl 和 best_model.pkl 到 saved_models 目录")
    st.stop()

# 加载模型
try:
    model = joblib.load('saved_models/best_model.pkl')
    scaler = joblib.load('saved_models/preprocessor.pkl')
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

# 创建界面
st.set_page_config(page_title="再手术风险预测", layout="wide")
st.title("再手术风险预测模型")

# 侧边栏输入
st.sidebar.header("患者信息")
inputs = {
    'Sex': st.sidebar.radio("性别 (0=女, 1=男)", [0, 1], index=0),
    'ASA scores': st.sidebar.slider("ASA评分", 0, 5, 2),
    'tumor location': st.sidebar.selectbox("肿瘤位置 (1-4)", [1, 2, 3, 4], index=1),
    'Benign or malignant': st.sidebar.radio("肿瘤性质 (0=良性, 1=恶性)", [0, 1], index=0),
    'Admitted to NICU': st.sidebar.radio("入住NICU (0=否, 1=是)", [0, 1], index=0),
    'Duration of surgery': st.sidebar.radio("手术时长 (0=<3小时, 1=≥3小时)", [0, 1], index=0),
    'diabetes': st.sidebar.radio("糖尿病史 (0=无, 1=有)", [0, 1], index=0),
    'CHF': st.sidebar.radio("心力衰竭 (0=无, 1=有)", [0, 1], index=0),
    'Functional dependencies': st.sidebar.radio("功能依赖 (0=无, 1=有)", [0, 1], index=0),
    'mFI-5': st.sidebar.slider("mFI-5评分", 0, 5, 1),
    'Type of tumor': st.sidebar.selectbox("肿瘤类型 (1-5)", [1, 2, 3, 4, 5], index=1)
}

# 创建输入数据框
input_df = pd.DataFrame([inputs])

# 显示输入数据
st.subheader("患者数据摘要")
st.dataframe(input_df)

# 预测按钮
if st.button("预测再手术风险"):
    try:
        # 预处理输入
        input_scaled = scaler.transform(input_df)
        
        # 预测概率
        risk = model.predict_proba(input_scaled)[0][1]
        
        # 显示结果
        st.subheader("预测结果")
        st.metric(label="再手术风险概率", value=f"{risk*100:.2f}%")
        
        # 风险可视化
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(['风险'], [risk], color='#ff6b6b' if risk > 0.5 else '#4ecdc4', height=0.5)
        ax.set_xlim(0, 1)
        ax.set_title('再手术风险概率')
        st.pyplot(fig)
        
        # 解释性分析
        st.subheader("风险解释")
        st.info("""
        高风险因素分析:
        - ASA评分越高风险越大
        - 恶性肿瘤比良性风险高
        - 手术时长超过3小时增加风险
        - 存在心力衰竭(CHF)显著增加风险
        """)
        
    except Exception as e:
        st.error(f"预测失败: {str(e)}")

# 底部信息
st.sidebar.header("关于")
st.sidebar.info("""
这是一个再手术风险预测模型。
输入患者信息后点击预测按钮获取结果。
""")