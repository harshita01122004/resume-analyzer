# app.py
# ─────────────────────────────────────────────────────────────────────────────
# ResumeIQ — Intelligent Resume Analysis System
# Pure Machine Learning Based — No ATS, No Gen AI
# Features: Login/Signup · Resume Parsing · Skill Extraction · ML Prediction
#           Resume Score · Skill Gap Analysis · Skills Analytics
# ─────────────────────────────────────────────────────────────────────────────

import re
from pathlib import Path

import streamlit as st
# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ — ML Resume Analyser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif; }

:root {
    --primary: #6366f1;
    --accent:  #06b6d4;
    --success: #10b981;
    --warning: #f59e0b;
    --danger:  #ef4444;
    --bg:      #07071a;
    --surface: #0f0f2a;
    --card:    #13132e;
    --card2:   #1a1a38;
    --border:  #252550;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --light:   #94a3b8;
}

#MainMenu, footer, header { visibility: hidden; }
.stApp { background: var(--bg) !important; color: var(--text); }
section[data-testid="stSidebar"] {
    background: #09091f !important;
    border-right: 1px solid var(--border);
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 4px; }

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-glass {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.grad-text {
    background: linear-gradient(135deg, #6366f1 0%, #06b6d4 60%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.badge {
    display: inline-block;
    background: linear-gradient(135deg, #4f46e5, #06b6d4);
    color: white; border-radius: 50px;
    padding: 0.22rem 0.85rem;
    font-size: 0.75rem; font-weight: 600; margin: 0.2rem;
}
.badge-warn {
    display: inline-block;
    background: rgba(245,158,11,0.15);
    border: 1px solid rgba(245,158,11,0.4);
    color: #fbbf24; border-radius: 50px;
    padding: 0.22rem 0.85rem;
    font-size: 0.75rem; font-weight: 600; margin: 0.2rem;
}
.badge-success {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.4);
    color: #34d399; border-radius: 50px;
    padding: 0.22rem 0.85rem;
    font-size: 0.75rem; font-weight: 600; margin: 0.2rem;
}

.section-title {
    font-size: 0.72rem; font-weight: 700; color: #6366f1;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 1rem;
}

.metric-tile {
    background: var(--card2); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.2rem 1rem; text-align: center;
    position: relative; overflow: hidden;
}
.metric-tile::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366f1, #06b6d4);
}
.metric-tile .val {
    font-size: 2.2rem; font-weight: 900;
    background: linear-gradient(135deg, #a5b4fc, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1;
}
.metric-tile .lbl {
    font-size: 0.72rem; color: var(--muted); margin-top: 0.3rem;
    font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;
}

.role-pill {
    background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
    color: white; border-radius: 50px; padding: 0.6rem 1.8rem;
    font-size: 1.15rem; font-weight: 800; display: inline-block;
    margin: 0.5rem 0; box-shadow: 0 4px 20px rgba(99,102,241,0.35);
}

.conf-wrap { margin: 0.5rem 0; }
.conf-label {
    display: flex; justify-content: space-between;
    font-size: 0.8rem; color: var(--light); margin-bottom: 4px; font-weight: 500;
}
.conf-track { background: #1e1e40; border-radius: 6px; height: 8px; overflow: hidden; }
.conf-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #6366f1, #06b6d4);
}

.score-val {
    font-size: 3.5rem; font-weight: 900;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;
}
.score-label {
    font-size: 0.8rem; color: var(--muted);
    letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.3rem; font-weight: 600;
}
.score-grade {
    margin-top: 0.6rem; padding: 0.3rem 1.2rem; border-radius: 50px;
    font-size: 0.82rem; font-weight: 700; letter-spacing: 0.05em;
    display: inline-block;
}

.info-row {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.55rem 0; border-bottom: 1px solid #1c1c3a; font-size: 0.88rem;
}
.info-row:last-child { border-bottom: none; }
.info-icon { font-size: 1rem; width: 1.4rem; text-align: center; color: #6366f1; }
.info-key { color: var(--muted); width: 80px; font-size: 0.8rem; font-weight: 500; }
.info-val { color: var(--text); font-weight: 600; flex: 1; word-break: break-all; }

.login-wrap {
    max-width: 440px; margin: 3rem auto;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 20px; padding: 2.8rem 2.2rem;
    box-shadow: 0 20px 60px rgba(99,102,241,0.15);
}
.login-logo { text-align: center; font-size: 3.5rem; margin-bottom: 0.4rem; }
.login-title {
    text-align: center; font-size: 1.7rem; font-weight: 900;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem;
}
.login-sub { text-align: center; color: var(--muted); font-size: 0.88rem; margin-bottom: 2rem; }
.login-hint { text-align: center; color: #3d3d6a; font-size: 0.78rem; margin-top: 1.2rem; }

.user-chip {
    background: linear-gradient(135deg, #1a1a40, #252550);
    border: 1px solid #3d3d6a; border-radius: 12px;
    padding: 0.9rem 1rem; text-align: center; margin-bottom: 1.2rem;
}

.prog-wrap { margin: 0.4rem 0 0.8rem 0; }
.prog-label { font-size: 0.78rem; color: var(--muted); margin-bottom: 3px; font-weight: 500; }
.prog-track { background: #1a1a3a; border-radius: 6px; height: 10px; overflow: hidden; }
.prog-fill { height: 100%; border-radius: 6px; }

.div-text { text-align: center; color: #2d2d55; font-size: 0.8rem; margin: 0.8rem 0; }

.page-header { padding: 2rem 0 1rem 0; text-align: center; }
.page-header h1 { font-size: 2.6rem; font-weight: 900; margin: 0 0 0.4rem 0; }
.page-header p { color: var(--muted); font-size: 1rem; margin: 0; }

.feat-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.5rem 1.2rem; text-align: center;
}
.feat-icon { font-size: 2.2rem; margin-bottom: 0.8rem; }
.feat-title { font-size: 0.95rem; font-weight: 700; color: #a5b4fc; margin-bottom: 0.4rem; }
.feat-desc { font-size: 0.82rem; color: var(--muted); line-height: 1.5; }

/* ML Info Box */
.ml-info {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    font-size: 0.82rem;
    color: var(--light);
    margin-bottom: 1rem;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0d0d28 !important; border: 1px solid #2d2d55 !important;
    color: var(--text) !important; border-radius: 10px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}
div[data-testid="stFileUploader"] {
    background: #0d0d28 !important;
    border: 2px dashed #2d2d55 !important;
    border-radius: 14px !important;
}
.stTabs [data-baseweb="tab-list"] { background: #0d0d28 !important; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 0.83rem !important; }
.stTabs [aria-selected="true"] { background: #6366f1 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)


# ── User DB ───────────────────────────────────────────────────────────────────
if "users_db" not in st.session_state:
    st.session_state.users_db = {"admin": "admin123", "demo": "demo123"}

for key, val in [("logged_in", False), ("username", ""), ("show_signup", False)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_auth_page():
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown('<div class="login-title">ResumeIQ</div>', unsafe_allow_html=True)

        if not st.session_state.show_signup:
            st.markdown('<div class="login-sub">Sign in to your account</div>', unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter username", key="li_user")
            password = st.text_input("Password", placeholder="Enter password", type="password", key="li_pass")
            if st.button("Sign In →", use_container_width=True, type="primary"):
                u, p = username.strip(), password.strip()
                if not u or not p:
                    st.error("Please fill in all fields.")
                elif u in st.session_state.users_db and st.session_state.users_db[u] == p:
                    st.session_state.logged_in = True
                    st.session_state.username  = u
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            st.markdown('<div class="div-text">── Don\'t have an account? ──</div>', unsafe_allow_html=True)
            if st.button("Create Account", use_container_width=True):
                st.session_state.show_signup = True
                st.rerun()
            st.markdown('<div class="login-hint">Demo → <b>demo</b> / <b>demo123</b></div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="login-sub">Create a new account</div>', unsafe_allow_html=True)
            nu = st.text_input("Choose Username", placeholder="e.g. john_doe", key="su_user")
            np = st.text_input("Choose Password", placeholder="Min. 6 characters", type="password", key="su_pass")
            nc = st.text_input("Confirm Password", placeholder="Repeat password", type="password", key="su_conf")
            if st.button("Register →", use_container_width=True, type="primary"):
                nu, np, nc = nu.strip(), np.strip(), nc.strip()
                if not nu or not np:
                    st.error("Please fill in all fields.")
                elif nu in st.session_state.users_db:
                    st.error("Username already exists.")
                elif len(np) < 6:
                    st.error("Password must be at least 6 characters.")
                elif np != nc:
                    st.error("Passwords do not match.")
                else:
                    st.session_state.users_db[nu] = np
                    st.success("Account created! Please sign in.")
                    st.session_state.show_signup = False
                    st.rerun()
            st.markdown('<div class="div-text">── Already have an account? ──</div>', unsafe_allow_html=True)
            if st.button("← Back to Sign In", use_container_width=True):
                st.session_state.show_signup = False
                st.rerun()



if not st.session_state.logged_in:
    show_auth_page()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS AFTER LOGIN
# ══════════════════════════════════════════════════════════════════════════════
try:
    from resume_parser import parse_resume
    from skills        import extract_skills, categorise_skills, skill_gap, ROLE_REQUIRED_SKILLS
    from model         import train_model, predict_job_role, model_exists
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def badge_html(lst):
    return " ".join(f'<span class="badge">{s}</span>' for s in lst)

def conf_bar_html(label, pct):
    return f"""<div class="conf-wrap">
        <div class="conf-label"><span>{label}</span><span>{pct:.1f}%</span></div>
        <div class="conf-track"><div class="conf-fill" style="width:{min(pct,100):.1f}%"></div></div>
    </div>"""

def prog_bar(label, value, max_val, color="#6366f1"):
    pct = min(value / max_val * 100, 100) if max_val else 0
    return f"""<div class="prog-wrap">
        <div class="prog-label">{label} &nbsp;<b style="color:#a5b4fc">{value}/{max_val}</b></div>
        <div class="prog-track"><div class="prog-fill" style="width:{pct:.0f}%;background:{color}"></div></div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="user-chip">
        <div style="font-size:1.6rem">👤</div>
        <div style="font-weight:800;color:#a5b4fc;font-size:0.95rem">{st.session_state.username}</div>
        <div style="color:#3d3d6a;font-size:0.72rem;margin-top:2px">● Online</div>
    </div>""", unsafe_allow_html=True)

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.rerun()

    st.divider()
    st.markdown("### 🤖 ML Model")

    st.markdown("""
    <div class="ml-info">
        <b>Algorithm:</b> Logistic Regression<br>
    </div>""", unsafe_allow_html=True)

    if model_exists():
        st.success("✅ Model ready")
    else:
        st.warning("⚠️ No trained model")

    if st.button("🔄 Train / Retrain Model", use_container_width=True):
        with st.spinner("Training Logistic Regression …"):
            try:
                res = train_model()
                st.success(f"✅ Accuracy: {res['accuracy']:.2%}")
                st.caption(f"CV Mean: {res['cv_mean']:.2%} ± {res['cv_std']:.2%}")
                st.caption(f"Samples: {res['n_samples']} | Features: {res['n_features']}")
                with st.expander("Classification Report"):
                    st.code(res["classification_report"])
            except Exception as exc:
                st.error(f"Error: {exc}")

    st.divider()
    st.markdown("### ⚙️ Display Options")
    show_gap   = st.toggle("Skill Gap Analysis",  value=True)
    show_chart = st.toggle("Skills Distribution", value=True)
    show_raw   = st.toggle("Show Extracted Text", value=False)

    st.divider()
    st.caption("ResumeIQ · ML-Powered · scikit-learn")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
    <h1 class="grad-text">📄 ResumeIQ</h1>
    <p>Machine Learning powered resume analysis · Logistic Regression</p>
</div>""", unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
uploaded = st.file_uploader("Upload Resume (PDF)", type=["pdf"],
                             help="Processed locally — no data stored.",
                             label_visibility="collapsed")

if not uploaded:
    c1, c2, c3 = st.columns(3)
    for col, (icon, title, desc) in zip([c1, c2, c3], [
        ("🔍", "Resume Parsing",     "Extracts text using NLP — NLTK tokenization, stopword removal & lemmatization."),
        ("🛠️", "Skill Extraction",  "Identifies 200+ tech & domain skills using regex pattern matching."),
        ("🎯", "ML Role Prediction", "TF-IDF vectorization + Logistic Regression predicts best matching job role."),
    ]):
        col.markdown(f"""<div class="feat-card">
            <div class="feat-icon">{icon}</div>
            <div class="feat-title">{title}</div>
            <div class="feat-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👆 Upload your PDF resume above to get started.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PARSE
# ══════════════════════════════════════════════════════════════════════════════
pdf_bytes = uploaded.read()
with st.spinner("🔍 Parsing resume with NLP pipeline …"):
    try:
        parsed = parse_resume(pdf_bytes)
    except ValueError as exc:
        st.error(f"❌ {exc}")
        st.stop()

raw_text   = parsed["raw_text"]
prep_text  = parsed["preprocessed_text"]
skills     = extract_skills(raw_text)
categories = categorise_skills(skills)


# ══════════════════════════════════════════════════════════════════════════════
# SEC 1 — CANDIDATE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 👤 Candidate Overview")
col_info, col_metrics = st.columns([1.2, 2])

with col_info:
    name     = parsed.get("name")     or "—"
    email    = parsed.get("email")    or "—"
    phone    = parsed.get("phone")    or "—"
    linkedin = parsed.get("linkedin") or "—"
    st.markdown(f"""<div class="card">
        <div class="section-title">Contact Information</div>
        <div class="info-row"><span class="info-icon">👤</span><span class="info-key">Name</span><span class="info-val">{name}</span></div>
        <div class="info-row"><span class="info-icon">📧</span><span class="info-key">Email</span><span class="info-val">{email}</span></div>
        <div class="info-row"><span class="info-icon">📞</span><span class="info-key">Phone</span><span class="info-val">{phone}</span></div>
        <div class="info-row"><span class="info-icon">🔗</span><span class="info-key">LinkedIn</span><span class="info-val">{linkedin}</span></div>
    </div>""", unsafe_allow_html=True)

with col_metrics:
    m1, m2, m3, m4 = st.columns(4)
    for col, (val, lbl) in zip([m1, m2, m3, m4], [
        (len(raw_text.split()), "Words"),
        (len(skills),           "Skills"),
        (len(categories),       "Categories"),
        (len(prep_text.split()),"Tokens"),
    ]):
        col.markdown(f"""<div class="metric-tile">
            <div class="val">{val}</div><div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SEC 2 — NLP PREPROCESSING INFO
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔬 NLP Preprocessing Pipeline Details"):
    p1, p2, p3, p4 = st.columns(4)
    steps = [
        ("1️⃣ Text Extraction", "pdfplumber extracts raw text from PDF pages."),
        ("2️⃣ Text Cleaning",   "URLs, non-ASCII & extra whitespace removed."),
        ("3️⃣ Tokenization",    "NLTK punkt tokenizer splits text into words."),
        ("4️⃣ Stopword Removal","Common English words (the, is, at) removed."),
    ]
    for col, (title, desc) in zip([p1, p2, p3, p4], steps):
        col.markdown(f"""<div class="card" style="text-align:center;padding:1rem">
            <div style="font-size:1.4rem">{title.split()[0]}</div>
            <div style="font-weight:700;color:#a5b4fc;font-size:0.82rem;margin:0.3rem 0">{' '.join(title.split()[1:])}</div>
            <div style="color:var(--muted);font-size:0.75rem">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SEC 3 — SKILLS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🛠️ Extracted Skills")
if skills:
    cat_icons = {
        "data_science_ml": "📊", "cloud": "☁️", "cybersecurity": "🔒",
        "mobile": "📱", "databases": "🗄️", "devops": "⚙️",
        "web_frameworks": "🌐", "programming_languages": "💻",
        "data_engineering": "🔧", "networking": "📡",
        "methodologies": "📋", "bi_analytics": "📈"
    }
    tab_labels = ["🔍 All Skills"] + [
        f"{cat_icons.get(c,'🔹')} {c.replace('_',' ').title()}" for c in categories
    ]
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        st.markdown(f'<div style="padding:0.6rem 0">{badge_html(skills)}</div>', unsafe_allow_html=True)
    for i, (cat, cat_skills) in enumerate(categories.items(), 1):
        with tabs[i]:
            st.markdown(f'<div style="padding:0.6rem 0">{badge_html(cat_skills)}</div>', unsafe_allow_html=True)
else:
    st.warning("No known skills detected. Resume may use non-standard terminology.")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SEC 4 — ML PREDICTION (Main Feature)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🎯 ML Job Role Prediction")
st.markdown("""<div class="ml-info">
    <b>How it works:</b> Resume text → NLP Preprocessing → TF-IDF Vectorization (5000 features, unigrams + bigrams)
    → Logistic Regression Classifier → Predicted Job Role with confidence scores.
</div>""", unsafe_allow_html=True)

prediction = None

if not model_exists():
    st.error("No trained model found. Click **Train / Retrain Model** in the sidebar first.")
else:
    with st.spinner("🤖 Running TF-IDF + Logistic Regression …"):
        try:
            prediction = predict_job_role(prep_text)
        except Exception as exc:
            st.error(f"Prediction error: {exc}")

if prediction:
    col_pred, col_chart = st.columns([1, 1.8])

    with col_pred:
        st.markdown(f"""<div class="card" style="text-align:center;padding:2rem 1.5rem">
            <div class="section-title" style="text-align:center">Predicted Role</div>
            <div class="role-pill">{prediction['predicted_role']}</div>
            <div style="color:var(--muted);margin-top:0.8rem;font-size:0.82rem">Model Confidence</div>
            <div style="font-size:2.4rem;font-weight:900;color:#10b981">{prediction['confidence']:.1f}%</div>
            <div style="color:var(--muted);font-size:0.75rem;margin-top:0.4rem">Logistic Regression Score</div>
        </div>""", unsafe_allow_html=True)

    with col_chart:
        bars = "".join(conf_bar_html(r["role"], r["confidence"]) for r in prediction["top_roles"])
        st.markdown(f"""<div class="card">
            <div class="section-title">Top 5 Role Predictions (Probability Scores)</div>
            {bars}
            <div style="color:var(--muted);font-size:0.75rem;margin-top:0.8rem">
                ℹ️ Scores are predict_proba() outputs from Logistic Regression model
            </div>
        </div>""", unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SEC 5 — SKILL GAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if show_gap and prediction:
    st.markdown("## 📉 Skill Gap Analysis")
    pred_role = prediction["predicted_role"]
    missing   = skill_gap(skills, pred_role)
    required  = ROLE_REQUIRED_SKILLS.get(pred_role, [])

    if required:
        matched  = [s for s in required if s in skills]
        coverage = len(matched) / len(required) * 100 if required else 0

        cg1, cg2 = st.columns(2)
        with cg1:
            st.markdown(f"""<div class="card">
                <div class="section-title">✅ Matched Skills</div>
                <div>{''.join(f'<span class="badge-success">{s}</span>' for s in matched)
                      if matched else '<span style="color:var(--muted)">None matched</span>'}</div>
            </div>""", unsafe_allow_html=True)
        with cg2:
            st.markdown(f"""<div class="card">
                <div class="section-title">❌ Missing Skills</div>
                <div>{''.join(f'<span class="badge-warn">{s}</span>' for s in missing)
                      if missing else '<span class="badge-success">All covered ✓</span>'}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="card-glass">
            <div class="section-title">Skill Coverage for — {pred_role}</div>
            {prog_bar("Coverage", len(matched), len(required), "#6366f1")}
            <div style="color:var(--muted);font-size:0.8rem;margin-top:0.5rem">
                You match <b style="color:#a5b4fc">{coverage:.0f}%</b> of required skills for
                <b style="color:#a5b4fc">{pred_role}</b>.
                {f'Add: <b style="color:#fbbf24">{", ".join(missing)}</b> to strengthen your profile.' if missing else ''}
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info(f"No skill requirements mapped for **{pred_role}** yet.")

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SEC 6 — SKILLS DISTRIBUTION CHART
# ══════════════════════════════════════════════════════════════════════════════
if show_chart and categories:
    st.markdown("## 📈 Skills Distribution")
    try:
        import pandas as pd
        chart_data = pd.DataFrame({
            "Category": [c.replace("_", " ").title() for c in categories],
            "Skills":   [len(v) for v in categories.values()],
        }).sort_values("Skills", ascending=False)
        st.bar_chart(chart_data.set_index("Category"), color="#6366f1")
    except Exception:
        pass
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SEC 7 — RAW TEXT
# ══════════════════════════════════════════════════════════════════════════════
if show_raw:
    with st.expander("📝 Extracted & Preprocessed Text"):
        t1, t2 = st.tabs(["Raw Text", "Preprocessed (ML Input)"])
        with t1:
            st.text_area("", raw_text[:5000], height=280, label_visibility="collapsed")
        with t2:
            st.text_area("", prep_text[:5000], height=280, label_visibility="collapsed",
                         help="This is the text fed into TF-IDF vectorizer after NLP preprocessing.")