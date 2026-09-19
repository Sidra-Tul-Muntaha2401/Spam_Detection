import html
import pickle
import re
from pathlib import Path

import streamlit as st

from scripts.preprocess import clean_text


st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"


st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,.12), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(168,85,247,.10), transparent 30%),
        #070b14;
    color: #f8fafc;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
.brand {
    display:flex; align-items:center; gap:15px; margin-bottom:8px;
}
.brand-icon {
    width:55px; height:55px; border-radius:16px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    font-size:27px; box-shadow:0 10px 35px rgba(99,102,241,.25);
}
.brand-title {font-size:31px; font-weight:800; letter-spacing:-1px;}
.brand-subtitle {color:#94a3b8; font-size:14px; margin-top:3px;}
.badge-container {display:flex; gap:8px; flex-wrap:wrap; margin:20px 0 30px;}
.badge {
    padding:7px 13px; border-radius:100px; font-size:12px; font-weight:600;
    border:1px solid rgba(148,163,184,.15); background:rgba(15,23,42,.75);
}
.badge-blue {color:#93c5fd}.badge-purple {color:#c4b5fd}.badge-green {color:#86efac}
.card {
    background:rgba(15,23,42,.72); border:1px solid rgba(148,163,184,.12);
    border-radius:20px; padding:24px; margin-bottom:20px;
    box-shadow:0 15px 45px rgba(0,0,0,.20);
}
.section-title {font-size:20px; font-weight:750; margin-bottom:5px;}
.section-description {color:#94a3b8; font-size:13px; margin-bottom:20px;}
.result-spam {
    background:rgba(127,29,29,.18); border:1px solid rgba(248,113,113,.35);
    border-radius:18px; padding:25px; text-align:center;
}
.result-ham {
    background:rgba(6,78,59,.18); border:1px solid rgba(52,211,153,.35);
    border-radius:18px; padding:25px; text-align:center;
}
.result-icon {font-size:40px; margin-bottom:8px;}
.result-title {font-size:25px; font-weight:800; margin-bottom:5px;}
.result-description {color:#94a3b8; font-size:13px;}
.stat-card {
    background:rgba(15,23,42,.75); border:1px solid rgba(148,163,184,.12);
    border-radius:16px; padding:18px; text-align:center;
}
.stat-value {font-size:25px; font-weight:800;}
.stat-label {color:#94a3b8; font-size:12px; margin-top:3px;}
.message-preview {
    background:#020617; border:1px solid #1e293b; border-radius:14px;
    padding:16px; color:#cbd5e1; font-size:14px; line-height:1.7; min-height:100px;
}
.token-spam {
    background:rgba(239,68,68,.15); color:#fca5a5;
    border:1px solid rgba(239,68,68,.25); padding:4px 7px;
    border-radius:6px; margin:2px; display:inline-block;
}
.token-normal {
    background:rgba(100,116,139,.10); color:#cbd5e1;
    padding:4px 7px; border-radius:6px; margin:2px; display:inline-block;
}
section[data-testid="stSidebar"] {background:#080d18; border-right:1px solid rgba(148,163,184,.10);}
.stButton > button {border-radius:10px; font-weight:600; border:1px solid rgba(148,163,184,.15);}
.footer {text-align:center; color:#64748b; font-size:12px; padding-top:30px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Missing vectorizer: {VECTORIZER_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


try:
    model, vectorizer = load_model()
    model_loaded = True
    model_error = ""
except Exception as exc:
    model_loaded = False
    model_error = str(exc)


with st.sidebar:
    st.markdown("## 🛡️ SpamShield AI")
    st.markdown("---")
    st.markdown("### About")
    st.write(
        "A supervised machine learning application that classifies "
        "SMS and email messages as Spam or Legitimate (Ham)."
    )
    st.markdown("### ML Pipeline")
    st.markdown(
        "**1. Input Text** → **2. Text Cleaning** → "
        "**3. TF-IDF** → **4. Logistic Regression** → "
        "**5. Prediction**"
    )
    st.markdown("---")
    if model_loaded:
        st.success("Model loaded successfully")
    else:
        st.error("Model loading failed")
        st.caption(model_error)


st.markdown("""
<div class="brand">
    <div class="brand-icon">🛡️</div>
    <div>
        <div class="brand-title">SpamShield AI</div>
        <div class="brand-subtitle">
            Machine Learning powered SMS & Email Spam Detection
        </div>
    </div>
</div>
<div class="badge-container">
    <div class="badge badge-blue">🧠 Machine Learning</div>
    <div class="badge badge-purple">📊 TF-IDF</div>
    <div class="badge badge-purple">🤖 Logistic Regression</div>
    <div class="badge badge-green">⚡ Real-Time Prediction</div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="card">
    <div class="section-title">🔍 Analyze a Message</div>
    <div class="section-description">
        Enter an SMS or email message and let the trained machine learning
        model classify it.
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("### Try an example")

c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("🎁 Prize Message", use_container_width=True):
        st.session_state["message"] = (
            "Congratulations! You have won a $1000 prize. "
            "Call now to claim your reward."
        )

with c2:
    if st.button("🚨 Urgent Alert", use_container_width=True):
        st.session_state["message"] = (
            "URGENT! Your account has been selected for a "
            "special cash reward. Click the link now."
        )

with c3:
    if st.button("☕ Normal Chat", use_container_width=True):
        st.session_state["message"] = (
            "Hey, are we still meeting for lunch today?"
        )

with c4:
    if st.button("💼 Work Message", use_container_width=True):
        st.session_state["message"] = (
            "Hi team, please review the project report before tomorrow's meeting."
        )


message = st.text_area(
    "Message",
    value=st.session_state.get("message", ""),
    height=170,
    placeholder="Type or paste your message here...",
    label_visibility="collapsed",
)

b1, b2 = st.columns([3, 1])

with b1:
    analyze = st.button("🔎 Analyze Message", type="primary", use_container_width=True)

with b2:
    clear = st.button("🗑️ Clear", use_container_width=True)

if clear:
    st.session_state["message"] = ""
    st.rerun()


if analyze:
    if not model_loaded:
        st.error("The ML model could not be loaded. Run the training script first.")
        st.stop()

    if not message.strip():
        st.warning("Please enter a message before analyzing.")
        st.stop()

    cleaned_message = clean_text(message)
    vectorized_message = vectorizer.transform([cleaned_message])
    prediction = model.predict(vectorized_message)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_message)[0]
        classes = list(model.classes_)
        spam_probability = (
            probabilities[classes.index(1)] if 1 in classes else 0.0
        )
    else:
        spam_probability = 1.0 if prediction == 1 else 0.0

    spam_percentage = spam_probability * 100
    ham_percentage = 100 - spam_percentage

    st.markdown("---")
    result_col, probability_col = st.columns([1, 1])

    with result_col:
        if prediction == 1:
            st.markdown(
                """
                <div class="result-spam">
                    <div class="result-icon">🚨</div>
                    <div class="result-title" style="color:#f87171;">
                        SPAM DETECTED
                    </div>
                    <div class="result-description">
                        The trained model classified this message as potentially unwanted.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-ham">
                    <div class="result-icon">✅</div>
                    <div class="result-title" style="color:#34d399;">
                        LEGITIMATE MESSAGE
                    </div>
                    <div class="result-description">
                        The trained model classified this message as legitimate.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with probability_col:
        st.markdown(
            '<div class="card"><div class="section-title">📊 Model Confidence</div>',
            unsafe_allow_html=True,
        )
        st.metric("Spam Probability", f"{spam_percentage:.1f}%")
        st.progress(int(round(spam_percentage)))
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Spam", f"{spam_percentage:.1f}%")
        with p2:
            st.metric("Ham", f"{ham_percentage:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## 🧪 Message Analysis")
    a1, a2 = st.columns(2)

    with a1:
        st.markdown(
            f"""
            <div class="card">
                <div class="section-title">📨 Original Message</div>
                <div class="message-preview">{html.escape(message)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            f"""
            <div class="card">
                <div class="section-title">🧹 Processed Text</div>
                <div class="message-preview">{html.escape(cleaned_message)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## ⚙️ Text Preprocessing Pipeline")

    original_tokens = message.lower().split()
    cleaned_tokens = cleaned_message.split()

    p1, p2, p3, p4 = st.columns(4)

    stats = [
        ("📝", "Raw Text"),
        (str(len(original_tokens)), "Original Words"),
        (str(len(cleaned_tokens)), "Cleaned Words"),
        (f"{vectorized_message.shape[1]:,}", "TF-IDF Features"),
    ]

    for col, (value, label) in zip((p1, p2, p3, p4), stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 🔤 Processed Tokens")

    token_html = ""
    for token in cleaned_tokens:
        if re.search(
            r"(free|win|winner|prize|cash|offer|urgent|claim|click|money|reward|congrat)",
            token,
            re.IGNORECASE,
        ):
            token_html += f'<span class="token-spam">{html.escape(token)}</span>'
        else:
            token_html += f'<span class="token-normal">{html.escape(token)}</span>'

    st.markdown(
        f'<div class="message-preview">{token_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("## 🧠 How the Prediction Works")

    s1, s2, s3 = st.columns(3)

    explanations = [
        ("01 — Clean", "The message is normalized and unnecessary characters are removed."),
        ("02 — Vectorize", "TF-IDF converts the cleaned text into numerical features."),
        ("03 — Classify", "Logistic Regression predicts Spam or Ham from those features."),
    ]

    for col, (title, description) in zip((s1, s2, s3), explanations):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="section-title">{title}</div>
                    <p style="color:#94a3b8;font-size:13px;">{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if not analyze and not message:
    st.markdown(
        """
        <div class="card" style="text-align:center;padding:45px;">
            <div style="font-size:45px;">🛡️</div>
            <div style="font-size:22px;font-weight:800;margin-top:10px;">
                Your inbox, protected by Machine Learning.
            </div>
            <div style="color:#94a3b8;margin-top:8px;font-size:14px;">
                Enter a message above to see how your trained ML model classifies it.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="footer">
        SpamShield AI • Python • Streamlit • TF-IDF • Logistic Regression
        <br><br>
        Machine Learning • NLP • Binary Classification
    </div>
    """,
    unsafe_allow_html=True,
)
