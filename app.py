import io, time, cv2, numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import plotly.express as px
import pandas as pd

st.set_page_config(page_title='Face Detection - WIDER FACE', page_icon='😊', layout='wide')

COMPARISON = {
    'Model': ['From Scratch', 'Transfer Learning', 'Fine-Tuning'],
    'mAP@0.5': [0.4547, 0.5532, 0.5899],
    'mAP@0.5:0.95': [0.2231, 0.2850, 0.3055],
    'Precision': [0.7227, 0.7920, 0.8010],
    'Recall': [0.4025, 0.4920, 0.5284],
    'Train(min)': [24.7, 90.1, 108.2],
}

@st.cache_resource
def load_model():
    return YOLO('best_model.pt')
model = load_model()

with st.sidebar:
    st.markdown('## Face Detector')
    st.markdown('**Dataset:** WIDER FACE')
    st.markdown('**Images:** 32,203 | **Boxes:** 393,703')
    st.markdown('---')
    conf = st.slider('Confidence', 0.05, 0.95, 0.30, 0.05)
    iou  = st.slider('IoU (NMS)',  0.10, 0.95, 0.45, 0.05)
    st.markdown('---')
    st.markdown('[Dataset on Kaggle](https://www.kaggle.com/datasets/mksaad/wider-face-a-face-detection-benchmark)')

st.title('Face Detection — WIDER FACE Dataset')
st.caption('YOLOv8n fine-tuned on 32,203 real-world images across 61 event categories.')

tab_det, tab_cmp, tab_about = st.tabs(['Detect', 'Model Comparison', 'About'])

with tab_det:
    up = st.file_uploader('Upload image', type=['jpg','jpeg','png','webp'])
    if up:
        img = Image.open(up).convert('RGB')
        arr = np.array(img)
        with st.spinner('Detecting faces ...'):
            t0  = time.perf_counter()
            res = model.predict(arr, conf=conf, iou=iou, verbose=False)[0]
            ms  = (time.perf_counter() - t0) * 1000
        boxes = res.boxes
        n     = len(boxes) if boxes else 0
        ann   = Image.fromarray(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Faces detected', n)
        c2.metric('Latency', f'{ms:.0f} ms')
        c3.metric('FPS', f'{1000/max(ms,1):.1f}')
        avg = np.mean([float(b.conf[0]) for b in boxes]) * 100 if n else 0
        c4.metric('Avg confidence', f'{avg:.0f}%')
        ca, cb = st.columns(2)
        ca.image(img, caption='Original',   use_container_width=True)
        cb.image(ann, caption='Detections', use_container_width=True)
        buf = io.BytesIO()
        ann.save(buf, format='JPEG')
        cb.download_button('Download result', buf.getvalue(), 'detected.jpg', 'image/jpeg')
        if n:
            st.markdown('### Detection Details')
            rows_det = []
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                rows_det.append({
                    '#': i+1,
                    'Score': f'{float(b.conf[0]):.1%}',
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'W': x2-x1, 'H': y2-y1,
                })
            st.dataframe(pd.DataFrame(rows_det), hide_index=True, use_container_width=True)
    else:
        st.info('Upload an image to detect faces.')

with tab_cmp:
    df_c = pd.DataFrame(COMPARISON)
    st.dataframe(df_c, hide_index=True, use_container_width=True)
    fig = px.bar(
        df_c, x='Model', y='mAP@0.5', color='Model',
        text='mAP@0.5', title='mAP@0.5 — WIDER FACE Validation Set'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, 0.75], height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    table_md = (
        '| Approach | Strategy | Key point |\n'
        '|---|---|---|\n'
        '| From Scratch | Random init | Baseline |\n'
        '| Transfer TL | Freeze 10 layers | Faster, moderate mAP |\n'
        '| Fine-Tuning | All layers, LR=1e-4 | Best mAP |'
    )
    st.markdown(table_md)

with tab_about:
    st.markdown('## About WIDER FACE')
    about_text = (
        '**WIDER FACE** is a large-scale face detection benchmark. '
        '32,203 images collected from 61 event categories '
        '(parade, wedding, sports, concert...). '
        'Faces span from tiny crowd faces to full-frame close-ups. '
        'Blur, occlusion, partial visibility, and label noise make it a realistic training set.\n\n'
        '**Stack:** PyTorch · Ultralytics YOLOv8 · OpenCV · Streamlit · Plotly'
    )
    st.markdown(about_text)
