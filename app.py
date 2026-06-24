"""
AI Video Assistant — Streamlit UI
A clean, professional front-end for the video summarization + RAG pipeline.
"""

import streamlit as st
from dotenv import load_dotenv
import os

from audio_processor import process_input
from transcriber import transcribe_all
from summarizer import summarize, generate_title
from insight_extractor import extract_action_items, extract_key_decisions, extract_questions
from rag_engine import build_rag_chain, ask_question

load_dotenv()

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="EchoNote · AI Video Assistant",
    page_icon="🪻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# DESIGN TOKENS — fixed palette, do not introduce other colors
# ----------------------------------------------------------------------------
C_LILAC      = "#9FA1FF"   # primary accent / links
C_PERIWINKLE = "#B5BAFF"   # secondary accent / borders
C_SKY        = "#AEE2FF"   # info surfaces
C_MINT       = "#D9F9DF"   # success / positive surfaces
C_VIOLET     = "#C9BEFF"   # tertiary surfaces / chips
C_INDIGO     = "#6367FF"   # deep accent / buttons / headings

INK      = "#1B1A3A"   # near-black ink for text on light surfaces
INK_SOFT = "#3A3960"   # secondary text
WHITE    = "#FFFFFF"
C_INPUT_TEXT = "#2A2EBF"   # blue text used inside input boxes for visibility

# ----------------------------------------------------------------------------
# GLOBAL CSS
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {INK};
}}

.stApp {{
    background: linear-gradient(160deg, #F7F8FF 0%, #EEF0FF 38%, #E8F8FF 100%);
}}

/* ---------- Hero ---------- */
.hero {{
    background: linear-gradient(120deg, {C_INDIGO} 0%, {C_LILAC} 55%, {C_SKY} 100%);
    border-radius: 22px;
    padding: 2.4rem 2.6rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 18px 40px -18px rgba(99,103,255,0.55);
    position: relative;
    overflow: hidden;
}}
.hero::after {{
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.18);
    border-radius: 50%;
}}
.hero h1 {{
    font-family: 'Sora', sans-serif;
    color: {WHITE};
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}}
.hero p {{
    color: rgba(255,255,255,0.92);
    font-size: 1.02rem;
    margin-top: 0.5rem;
    max-width: 640px;
}}
.hero .badge {{
    display: inline-block;
    background: rgba(255,255,255,0.22);
    color: {WHITE};
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.28rem 0.8rem;
    border-radius: 999px;
    margin-bottom: 0.8rem;
    letter-spacing: 0.3px;
}}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #6367FF 0%, #7d80ff 100%);
}}
section[data-testid="stSidebar"] * {{
    color: {WHITE} !important;
}}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
    background: {WHITE} !important;
    border: 1px solid rgba(255,255,255,0.5);
    color: {C_INPUT_TEXT} !important;
    border-radius: 10px;
    font-weight: 600;
}}
section[data-testid="stSidebar"] .stTextInput input::placeholder {{
    color: {C_INPUT_TEXT} !important;
    opacity: 0.55;
}}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div * {{
    color: {C_INPUT_TEXT} !important;
}}
section[data-testid="stSidebar"] .stRadio label, section[data-testid="stSidebar"] .stRadio div {{
    color: {WHITE} !important;
}}
section[data-testid="stSidebar"] label {{
    font-weight: 600;
    font-size: 0.85rem;
    color: {WHITE} !important;
}}
section[data-testid="stSidebar"] .stFileUploader section {{
    background: rgba(255,255,255,0.10);
    border: 1px dashed rgba(255,255,255,0.5);
    border-radius: 12px;
}}

/* ---------- Stepper (pipeline stages) ---------- */
.stepper {{
    display: flex;
    gap: 0.6rem;
    margin: 0.4rem 0 1.4rem 0;
    flex-wrap: wrap;
}}
.step {{
    flex: 1;
    min-width: 150px;
    background: {WHITE};
    border: 1px solid {C_PERIWINKLE};
    border-radius: 14px;
    padding: 0.7rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: {INK_SOFT};
    display: flex;
    align-items: center;
    gap: 0.55rem;
    transition: all 0.25s ease;
}}
.step .dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    background: {C_PERIWINKLE};
    flex-shrink: 0;
}}
.step.done {{
    border-color: {C_INDIGO};
    background: linear-gradient(120deg, #FFFFFF 70%, {C_MINT} 140%);
    color: {INK};
}}
.step.done .dot {{ background: {C_INDIGO}; }}
.step.active {{
    border-color: {C_INDIGO};
    box-shadow: 0 0 0 3px rgba(99,103,255,0.18);
    color: {INK};
}}
.step.active .dot {{
    background: {C_INDIGO};
    box-shadow: 0 0 0 4px rgba(99,103,255,0.25);
}}

/* ---------- Section cards ---------- */
.card {{
    background: {WHITE};
    border: 1px solid {C_PERIWINKLE};
    border-radius: 18px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 10px 26px -18px rgba(40,40,90,0.25);
}}
.card h3 {{
    font-family: 'Sora', sans-serif;
    color: {C_INDIGO};
    font-size: 1.08rem;
    margin-top: 0;
    margin-bottom: 0.7rem;
    font-weight: 700;
}}
.card p, .card li {{
    color: {INK};
    font-size: 0.97rem;
    line-height: 1.6;
}}
.title-pill {{
    display: inline-block;
    background: {C_MINT};
    color: {INK};
    font-weight: 700;
    font-family: 'Sora', sans-serif;
    padding: 0.55rem 1.1rem;
    border-radius: 12px;
    font-size: 1.15rem;
    margin-bottom: 1rem;
    border: 1px solid #BFEFC8;
}}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: rgba(255,255,255,0.5);
    padding: 6px;
    border-radius: 14px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    padding: 0.5rem 1.1rem;
    font-weight: 600;
    color: {INK_SOFT};
    background: transparent;
}}
.stTabs [aria-selected="true"] {{
    background: {C_INDIGO} !important;
    color: {WHITE} !important;
}}

/* ---------- Buttons ---------- */
div.stButton > button, div.stFormSubmitButton > button {{
    background: linear-gradient(120deg, {C_INDIGO}, {C_LILAC});
    color: {WHITE};
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.6rem 1.4rem;
    box-shadow: 0 10px 22px -10px rgba(99,103,255,0.7);
    transition: transform 0.15s ease;
}}
div.stButton > button:hover {{
    transform: translateY(-1px);
    color: {WHITE};
}}

/* ---------- Chat ---------- */
.chat-bubble-user {{
    background: {C_SKY};
    color: {INK};
    padding: 0.7rem 1rem;
    border-radius: 14px 14px 4px 14px;
    margin: 0.3rem 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
}}
.chat-bubble-bot {{
    background: {C_VIOLET};
    color: {INK};
    padding: 0.7rem 1rem;
    border-radius: 14px 14px 14px 4px;
    margin: 0.3rem 0;
    max-width: 80%;
    font-size: 0.95rem;
}}

/* ---------- Transcript ---------- */
.transcript-box {{
    background: {WHITE};
    border: 1px solid {C_PERIWINKLE};
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    max-height: 420px;
    overflow-y: auto;
    line-height: 1.8;
    font-size: 0.95rem;
    color: {INK};
}}
.transcript-box p {{
    margin-bottom: 0.9rem;
}}
.transcript-box::-webkit-scrollbar {{
    width: 8px;
}}
.transcript-box::-webkit-scrollbar-thumb {{
    background: {C_LILAC};
    border-radius: 8px;
}}

/* misc */
hr {{ border-color: {C_PERIWINKLE}; }}
::placeholder {{ color: {INK_SOFT} !important; opacity: 0.7; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
defaults = {
    "result": None,
    "rag_chain": None,
    "chat_history": [],
    "stage": -1,  # -1 = idle, 0..3 = pipeline stage in progress, 4 = done
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

STAGES = ["Transcribing", "Summarizing", "Extracting Insights", "Indexing for Chat"]

def render_stepper(active_idx: int):
    """active_idx: -1 idle, 0..3 current stage running, 4 = all done"""
    html = '<div class="stepper">'
    for i, label in enumerate(STAGES):
        if active_idx > i or active_idx == 4:
            cls = "step done"
        elif active_idx == i:
            cls = "step active"
        else:
            cls = "step"
        html += f'<div class="{cls}"><span class="dot"></span>{label}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SIDEBAR — inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🪻 EchoNote")
    st.caption("AI Video & Meeting Assistant")
    st.markdown("---")

    source_type = st.radio("Source type", ["YouTube URL", "Local file path"], label_visibility="visible")
    source = st.text_input(
        "Video / Audio source",
        placeholder="https://youtube.com/... or /path/to/file.mp4",
    )
    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    st.markdown("---")
    run_clicked = st.button("✨ Generate Summary", use_container_width=True)
    st.markdown("---")
    st.caption("Built with Streamlit · LangChain RAG · Whisper transcription")

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <span class="badge">AI-POWERED · MEETING & VIDEO INTELLIGENCE</span>
    <h1>Turn any video or meeting into clear, searchable knowledge.</h1>
    <p>Drop in a YouTube link or a local recording — EchoNote transcribes it, summarizes it,
    pulls out action items and decisions, and lets you chat with the content directly.</p>
</div>
""", unsafe_allow_html=True)

stepper_placeholder = st.empty()
with stepper_placeholder.container():
    render_stepper(st.session_state.stage)

# ----------------------------------------------------------------------------
# TRANSCRIPT FORMATTING HELPER
# ----------------------------------------------------------------------------
import re

def format_transcript(text: str) -> str:
    """Split a raw transcript into readable paragraphs of ~4 sentences each."""
    if not text:
        return "<p>No transcript available.</p>"
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    paragraphs, chunk = [], []
    for s in sentences:
        chunk.append(s)
        if len(chunk) >= 4:
            paragraphs.append(" ".join(chunk))
            chunk = []
    if chunk:
        paragraphs.append(" ".join(chunk))
    return "".join(f"<p>{p}</p>" for p in paragraphs)

# ----------------------------------------------------------------------------
# PIPELINE EXECUTION (with real per-stage progress)
# ----------------------------------------------------------------------------
if run_clicked:
    if not source:
        st.warning("Please enter a YouTube URL or a local file path before generating.")
    else:
        st.session_state.chat_history = []

        try:
            # Stage 0 — Transcribing
            st.session_state.stage = 0
            with stepper_placeholder.container():
                render_stepper(0)
            with st.spinner("Processing audio and transcribing…"):
                chunks = process_input(source)
                transcript = transcribe_all(chunks, language)

            # Stage 1 — Summarizing
            st.session_state.stage = 1
            with stepper_placeholder.container():
                render_stepper(1)
            with st.spinner("Generating title and summary…"):
                title = generate_title(transcript)
                summary = summarize(transcript)

            # Stage 2 — Extracting insights
            st.session_state.stage = 2
            with stepper_placeholder.container():
                render_stepper(2)
            with st.spinner("Extracting action items, decisions and questions…"):
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)

            # Stage 3 — Indexing for chat
            st.session_state.stage = 3
            with stepper_placeholder.container():
                render_stepper(3)
            with st.spinner("Building chat index…"):
                rag_chain = build_rag_chain(transcript)

            st.session_state.stage = 4
            with stepper_placeholder.container():
                render_stepper(4)

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
            }
            st.session_state.rag_chain = rag_chain
            st.success("Done! Explore the results below.")

        except Exception as e:
            st.session_state.stage = -1
            st.error(f"Something went wrong while processing your source: {e}")

# ----------------------------------------------------------------------------
# RESULTS
# ----------------------------------------------------------------------------
result = st.session_state.result

if result:
    st.markdown(f'<div class="title-pill">📌 {result["title"]}</div>', unsafe_allow_html=True)

    tabs = st.tabs(["📋 Summary", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "📝 Transcript", "💬 Chat"])

    with tabs[0]:
        st.markdown(f'<div class="card"><h3>Summary</h3><p>{result["summary"]}</p></div>', unsafe_allow_html=True)

    with tabs[1]:
        items = result["action_items"]
        items_html = "".join(f"<li>{i}</li>" for i in items) if isinstance(items, (list, tuple)) else f"<p>{items}</p>"
        st.markdown(f'<div class="card"><h3>Action Items</h3><ul>{items_html}</ul></div>', unsafe_allow_html=True)

    with tabs[2]:
        dec = result["key_decisions"]
        dec_html = "".join(f"<li>{d}</li>" for d in dec) if isinstance(dec, (list, tuple)) else f"<p>{dec}</p>"
        st.markdown(f'<div class="card"><h3>Key Decisions</h3><ul>{dec_html}</ul></div>', unsafe_allow_html=True)

    with tabs[3]:
        q = result["open_questions"]
        q_html = "".join(f"<li>{x}</li>" for x in q) if isinstance(q, (list, tuple)) else f"<p>{q}</p>"
        st.markdown(f'<div class="card"><h3>Open Questions</h3><ul>{q_html}</ul></div>', unsafe_allow_html=True)

    with tabs[4]:
        st.markdown(
            f'<div class="card"><h3>Full Transcript</h3>'
            f'<div class="transcript-box">{format_transcript(result["transcript"])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with tabs[5]:
        st.markdown('<div class="card"><h3>Chat with your meeting</h3></div>', unsafe_allow_html=True)

        for role, msg in st.session_state.chat_history:
            css_class = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
            label = "You" if role == "user" else "🤖 Assistant"
            st.markdown(f'<div class="{css_class}"><b>{label}:</b> {msg}</div>', unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            user_q = st.text_input("Ask a question about this video…", label_visibility="collapsed",
                                    placeholder="e.g. What did we decide about the launch date?")
            sent = st.form_submit_button("Send")

        if sent and user_q.strip():
            st.session_state.chat_history.append(("user", user_q.strip()))
            with st.spinner("Thinking…"):
                try:
                    answer = ask_question(st.session_state.rag_chain, user_q.strip())
                except Exception as e:
                    answer = f"Sorry, I couldn't answer that: {e}"
            st.session_state.chat_history.append(("bot", answer))
            st.rerun()

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem 2rem;">
        <h3 style="color:#6367FF;">No video processed yet</h3>
        <p>Enter a YouTube URL or local file path in the sidebar and click
        <b>"Generate Summary"</b> to get started.</p>
    </div>
    """, unsafe_allow_html=True)
