import streamlit as st
import whisper
import tempfile
import os
import time
import re
import threading

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Video Transcriber",
    page_icon="🎙️",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }

h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.4rem !important;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin-bottom: 0 !important;
}
h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #e0e0f0 !important; font-weight: 600 !important; }
p, label, .stMarkdown { color: #b0b0c8 !important; }

.stFileUploader > div {
    border: 2px dashed #7b2ff7 !important;
    border-radius: 16px !important;
    background: rgba(123, 47, 247, 0.05) !important;
    padding: 2rem !important;
    transition: all 0.3s ease;
}
.stFileUploader > div:hover {
    border-color: #00d4ff !important;
    background: rgba(0, 212, 255, 0.05) !important;
}

.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(123, 47, 247, 0.4) !important;
    border-radius: 10px !important;
    color: #e0e0f0 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7b2ff7, #00d4ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(123, 47, 247, 0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(123, 47, 247, 0.6) !important;
}

.stDownloadButton > button {
    background: rgba(0, 212, 255, 0.1) !important;
    color: #00d4ff !important;
    border: 1px solid #00d4ff !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}
.stDownloadButton > button:hover {
    background: rgba(0, 212, 255, 0.2) !important;
    transform: translateY(-1px) !important;
}

.stTextArea > div > div > textarea {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(123, 47, 247, 0.3) !important;
    border-radius: 12px !important;
    color: #e0e0f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.7 !important;
}

.stAlert { border-radius: 12px !important; border: none !important; }

[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(123, 47, 247, 0.2) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: #00d4ff !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #8888aa !important; }

hr { border-color: rgba(123, 47, 247, 0.2) !important; }
.stSpinner > div { border-top-color: #7b2ff7 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────

def format_timestamp_srt(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def build_txt(text: str) -> str:
    raw = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', raw)
    result = []
    for s in sentences:
        s = s.strip()
        if s:
            result.append(s)
    return "\n".join(result)


@st.cache_resource(show_spinner=False)
def load_model(model_size: str):
    return whisper.load_model(model_size)


# ── Session state init ────────────────────────────────────────
# Semua hasil transkripsi disimpan di session_state agar
# tidak hilang ketika tombol download dipencet (re-run).
for key in ("txt_output", "srt_output", "stats", "base_name"):
    if key not in st.session_state:
        st.session_state[key] = None


# ── UI ───────────────────────────────────────────────────────

st.title("🎙️ Video Transcriber")
st.markdown("**Transkripsi video otomatis** menggunakan Whisper — support Bahasa Indonesia, Inggris, dan Bali.")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")

    model_choice = st.selectbox(
        "Model Whisper",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=2,
        help="Semakin besar model → lebih akurat tapi lebih lambat.\n'small' direkomendasikan untuk video panjang di cloud.",
    )

    lang_choice = st.selectbox(
        "Bahasa",
        ["Bahasa Indonesia", "English", "Auto-Detect"],
        index=0,
    )
    lang_map = {"Bahasa Indonesia": "id", "English": "en", "Auto-Detect": None}
    lang_code = lang_map[lang_choice]

    st.markdown("---")
    st.markdown("#### ℹ️ Info Model")
    model_info = {
        "tiny":     ("~39M param",  "~1 GB RAM",  "⚡⚡⚡"),
        "base":     ("~74M param",  "~1 GB RAM",  "⚡⚡⚡"),
        "small":    ("~244M param", "~2 GB RAM",  "⚡⚡"),
        "medium":   ("~769M param", "~5 GB RAM",  "⚡"),
        "large-v3": ("~1.5B param", "~10 GB RAM", "🐢"),
    }
    info = model_info[model_choice]
    st.markdown(f"- **Ukuran:** {info[0]}")
    st.markdown(f"- **RAM:** {info[1]}")
    st.markdown(f"- **Kecepatan:** {info[2]}")
    st.markdown("---")
    st.markdown("<small style='color:#666'>Made with ❤️ + Whisper</small>", unsafe_allow_html=True)


# Upload
uploaded_file = st.file_uploader(
    "📁 Upload File Video",
    type=["mp4", "mkv", "avi", "mov", "webm", "flv", "m4v", "mpeg"],
    help="Format yang didukung: MP4, MKV, AVI, MOV, WEBM, FLV, M4V, MPEG",
)

if uploaded_file:
    file_mb = uploaded_file.size / (1024 * 1024)
    st.success(f"✅ **{uploaded_file.name}** ({file_mb:.1f} MB) berhasil diupload.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Nama File", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
    col2.metric("Ukuran", f"{file_mb:.1f} MB")
    col3.metric("Model", model_choice)

    st.markdown("")
    start_btn = st.button("🚀 Mulai Transkripsi", use_container_width=True)

    if start_btn:
        # Reset state hasil sebelumnya
        st.session_state.txt_output = None
        st.session_state.srt_output = None
        st.session_state.stats      = None
        st.session_state.base_name  = None

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner(f"⏳ Memuat model **{model_choice}**..."):
                model = load_model(model_choice)

            progress_bar = st.progress(0, text="🎧 Memulai transkripsi...")
            start_time   = time.time()

            result_holder = {}
            error_holder  = {}

            def run_transcribe():
                try:
                    result_holder["result"] = model.transcribe(
                        tmp_path,
                        language=lang_code,
                        task="transcribe",
                        verbose=False,
                        condition_on_previous_text=True,
                        fp16=False,
                        temperature=0.0,
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.6,
                        word_timestamps=False,
                    )
                except Exception as e:
                    error_holder["error"] = str(e)

            thread = threading.Thread(target=run_transcribe)
            thread.start()

            while thread.is_alive():
                elapsed = time.time() - start_time
                fake_progress = min(0.95, elapsed / (60 * 3))
                progress_bar.progress(fake_progress, text=f"🎙️ Sedang mentranskripsi... ({elapsed:.0f}s)")
                time.sleep(1)

            thread.join()

            if "error" in error_holder:
                st.error(f"❌ Gagal transkripsi: {error_holder['error']}")
            else:
                result        = result_holder["result"]
                elapsed_total = time.time() - start_time
                progress_bar.progress(1.0, text="✅ Transkripsi selesai!")

                # ── Simpan ke session_state ──────────────────────
                st.session_state.txt_output = build_txt(result["text"].strip())
                st.session_state.srt_output = build_srt(result["segments"])
                st.session_state.base_name  = os.path.splitext(uploaded_file.name)[0]
                st.session_state.stats      = {
                    "duration":      result["segments"][-1]["end"] if result["segments"] else 0,
                    "elapsed":       elapsed_total,
                    "segments":      len(result["segments"]),
                    "language":      result.get("language", "N/A"),
                }

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

else:
    # File di-clear → reset hasil agar tidak tampil stale
    st.session_state.txt_output = None
    st.session_state.srt_output = None
    st.session_state.stats      = None
    st.session_state.base_name  = None
    st.info("👆 Upload file video di atas untuk memulai transkripsi.")
    st.markdown("""
    **Cara penggunaan:**
    1. Upload file video (MP4, MKV, MOV, dll.)
    2. Pilih model dan bahasa di sidebar kiri
    3. Klik **Mulai Transkripsi**
    4. Tunggu proses selesai, lalu download hasil `.txt` atau `.srt`

    > 💡 **Tip:** Untuk video > 5 menit dengan Bahasa Indonesia, gunakan model **small** atau **medium**.
    """)


# ── Tampilkan hasil (dari session_state) ─────────────────────
# Blok ini selalu dirender ulang setiap re-run, tapi datanya
# tetap ada karena tersimpan di session_state — bukan di dalam
# blok if start_btn. Dengan begitu klik download tidak menghapus result.

if st.session_state.txt_output is not None:
    stats     = st.session_state.stats
    txt_output = st.session_state.txt_output
    srt_output = st.session_state.srt_output
    base_name  = st.session_state.base_name

    st.markdown("---")
    st.markdown("### 📊 Statistik")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Durasi Video",  f"{stats['duration']/60:.1f} mnt")
    c2.metric("Waktu Proses",  f"{stats['elapsed']:.0f} dtk")
    c3.metric("Segmen",        stats["segments"])
    c4.metric("Bahasa",        stats["language"].upper())

    st.markdown("---")
    st.markdown("### 📝 Hasil Transkripsi (.txt)")
    st.text_area(label="", value=txt_output, height=350, key="txt_preview")

    st.markdown("### 🎬 Subtitle (.srt)")
    st.text_area(label="", value=srt_output, height=250, key="srt_preview")

    st.markdown("---")
    st.markdown("### 💾 Download Hasil")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="⬇️ Download TXT",
            data=txt_output.encode("utf-8"),
            file_name=f"{base_name}_transkripsi.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            label="⬇️ Download SRT",
            data=srt_output.encode("utf-8"),
            file_name=f"{base_name}_subtitle.srt",
            mime="text/plain",
            use_container_width=True,
        )
