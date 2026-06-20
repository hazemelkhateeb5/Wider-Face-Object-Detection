
import io, os, time, cv2, numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import plotly.express as px, pandas as pd

st.set_page_config(page_title='Face Detection - WIDER FACE', page_icon='face', layout='wide')

MODEL_METRICS = {
    'Metric': ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'Train(min)'],
    'Value': [0.510, 0.285, 0.650, 0.510, 75],
}

# Resolve model path relative to this file so it works on Colab AND Streamlit Cloud
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.pt')

@st.cache_resource
def load(): return YOLO(MODEL_PATH)
model = load()

with st.sidebar:
    st.markdown('## Face Detector')
    st.markdown('**Dataset:** WIDER FACE')
    st.markdown('**Images:** 32,203 | **Boxes:** 393,703')
    st.markdown('---')
    conf = st.slider('Confidence', 0.05, 0.95, 0.30, 0.05)
    iou  = st.slider('IoU (NMS)',  0.10, 0.95, 0.45, 0.05)
    st.markdown('---')
    st.markdown('[Dataset on Kaggle](https://www.kaggle.com/datasets/mksaad/wider-face-a-face-detection-benchmark)')

st.title('Face Detection - WIDER FACE Dataset')
st.caption('YOLOv8n fine-tuned on 32,203 real-world images across 61 event categories.')

tab_det, tab_cmp, tab_about = st.tabs(['Detect', 'Model Metrics', 'About'])

with tab_det:
    up = st.file_uploader('Upload image', type=['jpg','jpeg','png','webp'])
    if up:
        img = Image.open(up).convert('RGB'); arr = np.array(img)
        with st.spinner('Detecting faces ...'):
            t0 = time.perf_counter()
            res = model.predict(arr, conf=conf, iou=iou, verbose=False)[0]
            ms = (time.perf_counter()-t0)*1000
        boxes = res.boxes; n = len(boxes) if boxes else 0
        ann = Image.fromarray(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB))
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Faces detected', n)
        c2.metric('Latency', f'{ms:.0f} ms')
        c3.metric('FPS', f'{1000/max(ms,1):.1f}')
        avg = np.mean([float(b.conf[0]) for b in boxes])*100 if n else 0
        c4.metric('Avg confidence', f'{avg:.0f}%')
        ca, cb = st.columns(2)
        ca.image(img,  caption='Original',   use_container_width=True)
        cb.image(ann,  caption='Detections', use_container_width=True)
        buf = io.BytesIO(); ann.save(buf, format='JPEG')
        cb.download_button('Download result', buf.getvalue(), 'detected.jpg', 'image/jpeg')
        if n:
            st.markdown('### Detection Details')
            rows_det = []
            for i, b in enumerate(boxes):
                x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
                rows_det.append({'#': i+1, 'Score': f'{float(b.conf[0]):.1%}',
                                 'x1':x1,'y1':y1,'x2':x2,'y2':y2,'W':x2-x1,'H':y2-y1})
            st.dataframe(pd.DataFrame(rows_det), hide_index=True, use_container_width=True)
    else:
        st.info('Upload an image to detect faces.')

with tab_cmp:
    df_m = pd.DataFrame(MODEL_METRICS)
    st.dataframe(df_m, hide_index=True, use_container_width=True)
    fig = px.bar(df_m, x='Metric', y='Value', text='Value',
                 title='YOLOv8n Fine-Tuned — WIDER FACE Validation Set')
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('**Strategy:** All layers unfrozen · AdamW · LR=1e-4 · mosaic+mixup · 20 epochs')

with tab_about:
    st.markdown('## About WIDER FACE')
    st.markdown('''
    **WIDER FACE** is a large-scale face detection benchmark.
    32,203 images collected from 61 event categories (parade, wedding, sports, concert...).
    Faces span from tiny crowd faces to full-frame close-ups.
    Blur, occlusion, partial visibility, and label noise make it a realistic training set.

    **Stack:** PyTorch - Ultralytics YOLOv8 - OpenCV - Streamlit - Plotly
    ''')
