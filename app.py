import numpy as np
import streamlit as st
import cv2
import pandas as pd
import urllib.parse
import time
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# --- 1. CORE ENGINE ---
st.set_page_config(page_title="VibeSync Elite", layout="wide", initial_sidebar_state="collapsed")

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_to_valence = {0: 0.15, 1: 0.20, 2: 0.25, 3: 0.95, 4: 0.50, 5: 0.10, 6: 0.85}

@st.cache_resource
def load_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.load_weights('model.h5')
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("muse_v3.csv")
    return df[['track', 'artist', 'valence_tags']].rename(columns={'track': 'name', 'valence_tags': 'pleasant'})

model = load_cnn()
df = load_data()

# --- 2. THE "ULTIMATE STUDIO" UI (PREMIUM CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syncopate:wght@700&family=Outfit:wght@300;600;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #050505 0%, #1a1a2e 50%, #000 100%);
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Neon Title */
    .hero-title {
        font-family: 'Syncopate';
        font-size: 7rem;
        text-align: center;
        background: linear-gradient(to right, #00f2fe, #4facfe, #7873f5, #ff3d77);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 30px rgba(79, 172, 254, 0.4));
        margin-top: -60px;
        letter-spacing: -10px;
    }

    /* Glassmorphism Song Strip */
    .music-strip {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 25px;
        padding: 25px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        transition: 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .music-strip:hover {
        background: rgba(255, 45, 85, 0.1);
        border-color: #ff2d55;
        transform: scale(1.05) rotate(1deg);
    }

    /* Pulsating Orb */
    .orb {
        width: 15px; height: 15px;
        background: #4facfe;
        border-radius: 50%;
        margin-right: 20px;
        box-shadow: 0 0 15px #4facfe;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0% { transform: scale(1); opacity: 1; } 100% { transform: scale(2.5); opacity: 0; } }

    /* Studio Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #111, #222) !important;
        border: 1px solid #444 !important;
        color: #fff !important;
        border-radius: 50px !important;
        padding: 20px !important;
        font-weight: 800 !important;
        font-family: 'Syncopate' !important;
        font-size: 0.6rem !important;
        letter-spacing: 3px;
        transition: 0.4s;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #00f2fe, #4facfe) !important;
        color: #000 !important;
        box-shadow: 0 0 40px rgba(0, 242, 254, 0.6) !important;
        transform: translateY(-5px);
    }

    .song-name { font-size: 1.4rem; font-weight: 800; color: #ffffff; }
    .artist-name { color: #ff2d55; font-size: 0.9rem; font-family: 'Syncopate'; }

    /* Visualizer Bars */
    .v-bar { width: 8px; background: #ff2d55; border-radius: 10px; animation: bounce 0.6s infinite alternate; }
    @keyframes bounce { from { height: 10px; } to { height: 60px; } }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER ---
st.markdown("<h1 class='hero-title'>VIBESYNC</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#444; letter-spacing:15px; font-size:0.8rem; margin-top:-30px;'>ADVANCED NEURAL AUDIO ARCHITECTURE</p>", unsafe_allow_html=True)

st.write("##")

col_left, col_right = st.columns([1, 2], gap="large")

# --- 4. CONSOLE PANEL (LEFT) ---
with col_left:
    st.write("### 🎚️ MASTER CONTROL")
    
    # SCAN TRIGGER
    if st.button("🔴 INITIATE NEURAL SCAN"):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        buffer = []
        feed = st.empty()
        
        for _ in range(40):
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi = np.expand_dims(np.expand_dims(cv2.resize(gray[y:y+h, x:x+w], (48, 48)), -1), 0).astype('float32') / 255.0
                pred = model.predict(roi, verbose=0)
                buffer.append(np.argmax(pred))
            feed.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=350)
        
        cap.release()
        feed.empty()
        
        if buffer:
            final_id = Counter(buffer).most_common(1)[0][0]
            st.session_state.emotion = emotion_dict[final_id]
            st.session_state.val = emotion_to_valence[final_id]
            st.rerun()

    st.write("##")
    
    # MOOD TILES
    m_row1 = st.columns(2)
    m_row2 = st.columns(2)
    m_row3 = st.columns(2)
    moods = ["Happy", "Sad", "Neutral", "Angry", "Surprised", "Fearful"]
    
    for i, m in enumerate(moods):
        target = [m_row1, m_row2, m_row3][i // 2]
        if target[i % 2].button(m):
            st.session_state.emotion = m
            rev_idx = {v: k for k, v in emotion_dict.items()}[m]
            st.session_state.val = emotion_to_valence[rev_idx]
            st.rerun()

    # Visualizer
    st.markdown("<div style='display:flex; justify-content:center; align-items:flex-end; height:80px; gap:8px; margin-top:30px;'><div class='v-bar'></div><div class='v-bar' style='animation-delay:0.2s; background:#4facfe;'></div><div class='v-bar' style='animation-delay:0.4s; background:#7873f5;'></div><div class='v-bar' style='animation-delay:0.1s;'></div></div>", unsafe_allow_html=True)

# --- 5. AUDIO OUTPUT (RIGHT) ---
with col_right:
    if 'emotion' in st.session_state:
        st.markdown(f"""
            <div style='margin-bottom:50px;'>
                <p style='color:#ff2d55; font-family:Syncopate; font-size:0.8rem; margin:0;'>OUTPUT FREQUENCY LOCKED</p>
                <h2 style='font-family:Syncopate; font-size:4.5rem; margin:0; line-height:1;'>{st.session_state.emotion.upper()}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        recs = df.iloc[(df['pleasant'] - st.session_state.val).abs().argsort()[:10]]
        
        grid_a, grid_b = st.columns(2)
        for i, (idx, row) in enumerate(recs.iterrows()):
            q = urllib.parse.quote(f"{row['name']} {row['artist']}")
            with (grid_a if i % 2 == 0 else grid_b):
                st.markdown(f"""
                    <div class="music-strip">
                        <div class="orb"></div>
                        <div>
                            <span class="song-name">{row['name']}</span><br>
                            <span class="artist-name">{row['artist']}</span><br>
                            <a href="https://www.youtube.com/results?search_query={q}" target="_blank" style="color:#00f2fe; text-decoration:none; font-weight:800; font-size:0.7rem;">▶ STREAM NOW</a>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='height:400px; display:flex; align-items:center; justify-content:center; border:1px solid #222; border-radius:30px; color:#333; font-family:Syncopate;'>SYSTEM STANDBY...</div>", unsafe_allow_html=True)