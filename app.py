from __future__ import annotations

import html
import tempfile
from datetime import datetime
from pathlib import Path

import altair as alt
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.audio import (
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    limit_audio_duration,
    load_audio_file,
)
from core.diarization import detect_speech_segments
from core.history import (
    add_history_entry,
    clear_history,
    delete_history_by_filename,
    delete_history_entry,
    load_history,
)
from core.inference import AnalysisResult, analyze_audio
from core.model import (
    CheckpointError,
    ModelBundle,
    clear_model_cache,
    discover_models,
    load_checkpoint,
)


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
STYLES_PATH = APP_DIR / "assets" / "styles.css"
DATA_DIR = APP_DIR / "data"
MAX_FILE_SIZE_MB = 100
MAX_ANALYSIS_SEC = 10
DIARIZATION_SR = 16000
NAV_PAGES = {
    "home": "🏠  หน้าแรก / วิเคราะห์เสียง",
    "history": "📋  ประวัติการวิเคราะห์",
    "files": "📁  จัดการไฟล์เสียง",
    "diarization": "👥  Speaker Diarization",
    "settings": "⚙️  ตั้งค่า",
}
MODEL_DISPLAY_NAMES = {
    "LFCC-TrainTest_eng2.2+thai2.2.pt": (
        "โมเดล — English 2.2 + Thai 2.2 (แบ่งตามโฟลเดอร์เดิม)"
    ),
    "LFCC-TrainTest_eng2.2+thai2.2_spkdisjoint.pt": (
        "โมเดล — English 2.2 + Thai 2.2 (speaker-disjoint: train 4 คน / val 2 คน)"
    ),
}
MODEL_SHORT_NAMES = {
    "LFCC-TrainTest_eng2.2+thai2.2.pt": "โฟลเดอร์เดิม",
    "LFCC-TrainTest_eng2.2+thai2.2_spkdisjoint.pt": "Speaker-disjoint",
}


def apply_styles() -> None:
    css = STYLES_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_html(markup: str) -> None:
    st.html(markup, width="stretch")


def format_metric(value: float | None, percentage: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%" if percentage else f"{value:.4f}"


def format_model_name(path: Path) -> str:
    return MODEL_DISPLAY_NAMES.get(path.name, path.stem)


def format_model_short(path: Path) -> str:
    return MODEL_SHORT_NAMES.get(path.name, path.stem)


def init_session_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "managed_files" not in st.session_state:
        st.session_state.managed_files = {}


def render_sidebar_brand() -> None:
    with st.sidebar:
        render_html(
            """
            <div class="brand">
                <div class="brand-mark">🎙️</div>
                <div>
                    <div class="brand-kicker">AI-Powered Analysis</div>
                    <div class="brand-title">Voice Authenticity Detector</div>
                    <div class="brand-sub">ตรวจจับเสียงจริงและเสียงปลอม</div>
                </div>
            </div>
            """
        )


def render_sidebar_nav() -> str:
    with st.sidebar:
        render_html('<div class="side-label">เมนูหลัก</div>')
        for key, label in NAV_PAGES.items():
            is_active = st.session_state.page == key
            if st.button(
                label,
                key=f"nav_{key}",
                width="stretch",
                type="primary" if is_active else "secondary",
            ):
                st.session_state.page = key
                st.rerun()
    return st.session_state.page


def render_sidebar_models(bundles: list[ModelBundle]) -> None:
    with st.sidebar:
        render_html('<div class="side-label">โมเดลที่พร้อมใช้</div>')
        if st.button("รีเฟรชรายการโมเดล", width="stretch", key="refresh_models"):
            clear_model_cache()
            st.rerun()
        render_html(f'<div class="side-status">พร้อมใช้งาน {len(bundles)} โมเดล</div>')
        for bundle in bundles:
            with st.expander(format_model_short(bundle.path)):
                config = bundle.config
                st.caption(format_model_name(bundle.path))
                st.caption(f"อุปกรณ์: `{bundle.device}`")
                st.write(f"Sample rate: `{config.sample_rate:,} Hz`")
                st.write(f"ช่วงต่อ segment: `{config.duration:g} วินาที`")
                st.write(f"LFCC: `{config.n_lfcc} × {config.frame_count}`")
                if bundle.metrics:
                    test_acc = bundle.metrics.get("test_acc")
                    val_acc = bundle.metrics.get("validation_combined_acc")
                    if test_acc is not None:
                        st.write(f"Test accuracy: `{format_metric(test_acc, True)}`")
                    elif val_acc is not None:
                        st.write(f"Val accuracy: `{format_metric(val_acc, True)}`")
        render_html(
            f"""
            <div class="side-hint">
                <strong>วิธีใช้</strong><br>
                เลือกเมนูด้านซ้ายเพื่อสลับหน้า · อัปโหลดไฟล์แล้วกดวิเคราะห์
                ระบบจะเทียบผลจากทุกโมเดลในครั้งเดียว
            </div>
            <div class="side-footer">
                รองรับ WAV · MP3 · FLAC · M4A · MP4<br>
                สูงสุด {MAX_FILE_SIZE_MB} MB · วิเคราะห์ {MAX_ANALYSIS_SEC} วินาทีแรก<br>
                © 2026 Voice Authenticity Detector
            </div>
            """
        )


def load_models(required: bool = True) -> list[ModelBundle]:
    models = discover_models(MODELS_DIR)
    if not models:
        if required:
            st.error(
                "ยังไม่พบโมเดล กรุณาวางไฟล์ `.pt` ในโฟลเดอร์ "
                f"`{MODELS_DIR}` แล้วกดรีเฟรช"
            )
            st.stop()
        return []

    bundles: list[ModelBundle] = []
    for model_path in models:
        try:
            bundles.append(load_checkpoint(model_path))
        except (CheckpointError, FileNotFoundError) as exc:
            if required:
                st.error(f"โหลดโมเดล `{model_path.name}` ไม่สำเร็จ: {exc}")
                st.stop()
    return bundles


def remember_uploaded_file(uploaded_file) -> None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.session_state.managed_files[uploaded_file.name] = {
        "name": uploaded_file.name,
        "size_mb": round(file_size_mb, 2),
        "suffix": Path(uploaded_file.name).suffix.lower(),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_analysis_history(
    uploaded_file,
    results: dict[str, AnalysisResult],
    bundles: list[ModelBundle],
) -> None:
    remember_uploaded_file(uploaded_file)
    model_summaries = []
    for bundle in bundles:
        result = results[bundle.path.name]
        model_summaries.append(
            {
                "model_file": bundle.path.name,
                "model_name": format_model_name(bundle.path),
                "label": result.label,
                "real_pct": round(result.real_probability * 100, 2),
                "fake_pct": round(result.fake_probability * 100, 2),
                "confidence_pct": round(result.confidence * 100, 2),
                "duration_sec": round(result.duration_sec, 2),
                "segments": len(result.segments),
            }
        )
    add_history_entry(
        {
            "file_name": uploaded_file.name,
            "file_size_mb": round(uploaded_file.size / (1024 * 1024), 2),
            "models": model_summaries,
        }
    )


def result_rows(result: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ช่วง": segment.index,
                "เริ่ม (วิ)": round(segment.start_sec, 2),
                "สิ้นสุด (วิ)": round(segment.end_sec, 2),
                "ผล": segment.label.upper(),
                "Real (%)": round(segment.real_probability * 100, 2),
                "Fake (%)": round(segment.fake_probability * 100, 2),
                "ความมั่นใจ (%)": round(segment.confidence * 100, 2),
                "RMS": round(segment.rms, 6),
            }
            for segment in result.segments
        ]
    )


def render_trend_chart(rows: pd.DataFrame) -> None:
    melted = rows.melt(
        id_vars=["เริ่ม (วิ)"],
        value_vars=["Real (%)", "Fake (%)"],
        var_name="ชนิด",
        value_name="เปอร์เซ็นต์",
    )
    melted["ชนิด"] = melted["ชนิด"].map({"Real (%)": "Real", "Fake (%)": "Fake"})
    chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(filled=True, size=55))
        .encode(
            x=alt.X("เริ่ม (วิ):Q", title="เวลาเริ่ม (วินาที)"),
            y=alt.Y(
                "เปอร์เซ็นต์:Q",
                title="ความมั่นใจ (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "ชนิด:N",
                scale=alt.Scale(
                    domain=["Real", "Fake"],
                    range=["#10B981", "#EF4444"],
                ),
                legend=alt.Legend(orient="top", title=None, symbolType="stroke"),
            ),
            tooltip=["เริ่ม (วิ)", "ชนิด", "เปอร์เซ็นต์"],
        )
        .properties(height=280)
        .configure(background="transparent")
        .configure_axis(
            gridColor="#e2e8f0",
            domainColor="#cbd5e1",
            tickColor="#cbd5e1",
            labelColor="#64748b",
            titleColor="#64748b",
        )
        .configure_view(strokeWidth=0)
        .configure_legend(labelColor="#475569")
    )
    st.altair_chart(chart, width="stretch")


def confidence_donut_style(confidence_pct: float, is_fake: bool) -> str:
    color = "#ef4444" if is_fake else "#10b981"
    return (
        f"background: conic-gradient({color} 0% {confidence_pct:.1f}%, "
        f"#e2e8f0 {confidence_pct:.1f}% 100%);"
    )


def waveform_spans(count: int = 12) -> str:
    return "".join("<span>&nbsp;</span>" for _ in range(count))


@st.cache_data(show_spinner=False)
def _load_audio_for_spectrogram(
    file_bytes: bytes,
    suffix: str,
    sample_rate: int,
    max_sec: float,
) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_path = Path(temp_file.name)
    try:
        return limit_audio_duration(
            load_audio_file(
                temp_path,
                sample_rate,
                max_duration_sec=max_sec,
            ),
            sample_rate,
            max_sec,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def render_spectrogram(
    uploaded_file,
    sample_rate: int = 16000,
) -> None:
    render_html(
        """
        <div class="section-head">
            <h3>Spectrogram</h3>
            <p>Spectrogram ของช่วงเสียงที่ใช้วิเคราะห์ (สูงสุด 10 วินาทีแรก)</p>
        </div>
        """
    )
    try:
        audio = _load_audio_for_spectrogram(
            uploaded_file.getvalue(),
            Path(uploaded_file.name).suffix.lower() or ".wav",
            sample_rate,
            MAX_ANALYSIS_SEC,
        )
        stft = np.abs(
            librosa.stft(audio, n_fft=1024, hop_length=256),
        )
        stft_db = librosa.amplitude_to_db(stft, ref=np.max)

        fig, ax = plt.subplots(figsize=(11, 3.4), dpi=120)
        image = librosa.display.specshow(
            stft_db,
            sr=sample_rate,
            hop_length=256,
            n_fft=1024,
            x_axis="time",
            y_axis="hz",
            cmap="magma",
            ax=ax,
        )
        ax.set_title(
            f"Spectrogram — {uploaded_file.name}",
            fontsize=11,
            pad=10,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hz")
        fig.colorbar(image, ax=ax, format="%+2.0f dB", pad=0.02)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True, width="stretch")
        plt.close(fig)
        st.caption(
            f"แสดงช่วง {len(audio) / sample_rate:.2f} วินาที · "
            f"sample rate {sample_rate:,} Hz"
        )
    except Exception as exc:
        st.warning(f"สร้าง Spectrogram ไม่สำเร็จ: {exc}")


def render_result(
    result: AnalysisResult,
    file_name: str,
    model_name: str,
    model_key: str,
) -> None:
    real_pct = result.real_probability * 100
    fake_pct = result.fake_probability * 100
    confidence_pct = result.confidence * 100
    is_fake = result.label == "fake"
    result_class = "fake" if is_fake else "real"
    result_text = "FAKE" if is_fake else "REAL"
    result_sub = "เสียงปลอม (Synthetic Voice)" if is_fake else "เสียงจริง (Authentic Voice)"
    shield_icon = "✕" if is_fake else "✓"
    confidence_caption = (
        "มั่นใจว่าเป็นเสียงปลอม"
        if is_fake
        else "มั่นใจว่าเป็นเสียงจริง"
    )
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")

    render_html(
        f"""
        <div class="results-section">
            <div class="results-title">ผลการวิเคราะห์ — {html.escape(model_name)}</div>
            <div class="results-grid">
                <div class="result-card">
                    <div class="result-card-label">ผลรวม</div>
                    <div class="verdict-shield {result_class}">{shield_icon}</div>
                    <div class="verdict-big {result_class}">{result_text}</div>
                    <div class="verdict-sub">{result_sub}</div>
                </div>
                <div class="result-card">
                    <div class="result-card-label">คะแนนความมั่นใจ</div>
                    <div class="confidence-donut" style="{confidence_donut_style(confidence_pct, is_fake)}">
                        <div class="confidence-donut-inner">
                            <div class="confidence-pct {result_class}">{confidence_pct:.0f}%</div>
                        </div>
                    </div>
                    <div class="confidence-caption">{confidence_caption}</div>
                </div>
                <div class="result-card">
                    <div class="result-card-label">ความน่าจะเป็น</div>
                    <div class="prob-card-body">
                        <div class="prob-list">
                            <div class="prob-row">
                                <div class="prob-name real">Real</div>
                                <div class="prob-track">
                                    <div class="prob-fill real" style="width:{real_pct:.1f}%"></div>
                                </div>
                                <div class="prob-pct">{real_pct:.1f}%</div>
                            </div>
                            <div class="prob-row">
                                <div class="prob-name fake">Fake</div>
                                <div class="prob-track">
                                    <div class="prob-fill fake" style="width:{fake_pct:.1f}%"></div>
                                </div>
                                <div class="prob-pct">{fake_pct:.1f}%</div>
                            </div>
                        </div>
                        <div class="waveform-bar" aria-hidden="true">{waveform_spans()}</div>
                    </div>
                </div>
                <div class="result-card">
                    <div class="result-card-label">รายละเอียดการวิเคราะห์</div>
                    <ul class="details-list">
                        <li><span class="dk">โมเดล</span><span class="dv">LFCC + CNN</span></li>
                        <li><span class="dk">ชื่อไฟล์</span><span class="dv">{html.escape(file_name)}</span></li>
                        <li><span class="dk">ความยาว</span><span class="dv">{result.duration_sec:.1f} วินาที</span></li>
                        <li><span class="dk">ช่วงที่วิเคราะห์</span><span class="dv">{len(result.segments)} ช่วง</span></li>
                        <li><span class="dk">เวลา</span><span class="dv">{timestamp}</span></li>
                    </ul>
                </div>
            </div>
        </div>
        """
    )
    st.caption(
        "ผลรวมคือ softmax เฉลี่ยแบบถ่วงน้ำหนักตามความยาวจริงของแต่ละช่วง "
        "และไม่ใช่ probability ที่ผ่านการ calibration"
    )

    rows = result_rows(result)
    render_html(
        """
        <div class="section-head">
            <h3>แนวโน้มรายช่วง</h3>
            <p>สัดส่วน Real / Fake ของแต่ละช่วงเสียงตามเวลา</p>
        </div>
        """
    )
    render_trend_chart(rows)

    render_html(
        """
        <div class="section-head">
            <h3>รายละเอียดรายช่วง</h3>
            <p>ดูค่าความน่าจะเป็นและความมั่นใจของทุกช่วงที่โมเดลวิเคราะห์</p>
        </div>
        """
    )
    st.dataframe(rows, width="stretch", hide_index=True)
    st.download_button(
        "💾 บันทึกผล (CSV)",
        data=rows.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{Path(file_name).stem}_{Path(model_key).stem}_analysis.csv",
        mime="text/csv",
        width="stretch",
        key=f"download_{model_key}",
    )


def analyze_upload(
    uploaded_file,
    bundles: list[ModelBundle],
) -> dict[str, AnalysisResult]:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = Path(temp_file.name)
    try:
        audio_by_sample_rate = {
            sample_rate: limit_audio_duration(
                load_audio_file(
                    temp_path,
                    sample_rate,
                    max_duration_sec=MAX_ANALYSIS_SEC,
                ),
                sample_rate,
                MAX_ANALYSIS_SEC,
            )
            for sample_rate in {bundle.config.sample_rate for bundle in bundles}
        }
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        bundle.path.name: analyze_audio(
            audio_by_sample_rate[bundle.config.sample_rate],
            bundle,
        )
        for bundle in bundles
    }


def render_comparison(
    results: dict[str, AnalysisResult],
    bundles: list[ModelBundle],
) -> None:
    render_html(
        f"""
        <div class="section-head">
            <h3>สรุปผลจากทั้ง {len(bundles)} โมเดล</h3>
            <p>เปรียบเทียบคำตอบหลักของแต่ละโมเดลจากไฟล์เดียวกัน</p>
        </div>
        """
    )
    cards = []
    for bundle in bundles:
        result = results[bundle.path.name]
        is_fake = result.label == "fake"
        result_class = "fake" if is_fake else "real"
        label = "เสียงปลอม (FAKE)" if is_fake else "เสียงจริง (REAL)"
        cards.append(
            '<div class="compare-card '
            f'{result_class}">'
            f'<div class="compare-model">{html.escape(format_model_name(bundle.path))}</div>'
            f'<div class="compare-title">{label}</div>'
            '<div class="compare-stats">'
            f"<span>Real <b>{result.real_probability * 100:.2f}%</b></span>"
            f"<span>Fake <b>{result.fake_probability * 100:.2f}%</b></span>"
            "</div></div>"
        )
    render_html(f'<div class="compare-grid">{"".join(cards)}</div>')


def render_hero(model_count: int) -> None:
    render_html(
        f"""
        <div class="topbar">
            <div class="topbar-btn" title="โหมดสว่าง">☀️</div>
            <div class="topbar-user">
                <div class="topbar-avatar">U</div>
                ผู้ใช้งาน ▾
            </div>
        </div>
        <div class="hero">
            <div class="hero-top">
                <div>
                    <h2>ตรวจสอบเสียงจริงและเสียงปลอม</h2>
                    <p>อัปโหลดไฟล์เสียงเพื่อให้โมเดลทั้ง {model_count} ตัววิเคราะห์พร้อมกัน
                    จากคุณลักษณะ LFCC + CNN บนเครื่องของคุณ</p>
                </div>
                <div class="hero-wave" aria-hidden="true">
                    <svg viewBox="0 0 180 60" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M0 30 Q15 10 30 30 T60 30 T90 30 T120 30 T150 30 T180 30"
                              stroke="#8b5cf6" stroke-width="2" fill="none" opacity="0.5"/>
                        <path d="M0 35 Q20 15 40 35 T80 35 T120 35 T160 35 T180 35"
                              stroke="#3b82f6" stroke-width="1.5" fill="none" opacity="0.35"/>
                    </svg>
                </div>
            </div>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🧠</div>
                    <div>
                        <div class="feature-title">LFCC + CNN</div>
                        <div class="feature-desc">สกัดคุณลักษณะ LFCC แล้วจำแนกด้วย CNN</div>
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div>
                        <div class="feature-title">วิเคราะห์ {model_count} โมเดล</div>
                        <div class="feature-desc">เทียบผลจากทุกโมเดลในครั้งเดียว</div>
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <div>
                        <div class="feature-title">CPU (เร็ว)</div>
                        <div class="feature-desc">ประมวลผลบนเครื่องของคุณโดยตรง</div>
                    </div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🎵</div>
                    <div>
                        <div class="feature-title">รองรับหลายรูปแบบ</div>
                        <div class="feature-desc">WAV · MP3 · FLAC · M4A · MP4</div>
                    </div>
                </div>
            </div>
        </div>
        """
    )


def render_file_preview_placeholder() -> None:
    render_html(
        """
        <div class="file-preview-card preview-empty">
            <div class="file-preview-icon">🎧</div>
            <div class="file-preview-info">
                <div class="file-preview-name">ยังไม่ได้เลือกไฟล์</div>
                <div class="file-preview-meta">อัปโหลดไฟล์เสียงเพื่อดูตัวอย่างและ waveform</div>
                <div class="waveform-bar muted" aria-hidden="true">"""
        + waveform_spans()
        + """</div>
            </div>
        </div>
        """
    )


def render_upload_header() -> None:
    render_html(
        f"""
        <div class="upload-panel-head">
            <h3>อัปโหลดไฟล์เสียง</h3>
            <p>ลากไฟล์มาวาง หรือเลือกไฟล์ · รองรับ WAV, MP3, FLAC, M4A, MP4
            · สูงสุด {MAX_FILE_SIZE_MB} MB · วิเคราะห์ {MAX_ANALYSIS_SEC} วินาทีแรก</p>
        </div>
        """
    )


def render_file_preview(file_name: str, file_size_mb: float, model_count: int) -> None:
    render_html(
        f"""
        <div class="file-preview-card">
            <div class="file-preview-icon">🎵</div>
            <div class="file-preview-info">
                <div class="file-preview-name">{html.escape(file_name)}</div>
                <div class="file-preview-meta">
                    {file_size_mb:.2f} MB · วิเคราะห์ {MAX_ANALYSIS_SEC} วินาทีแรก · {model_count} โมเดล
                </div>
                <div class="waveform-bar" aria-hidden="true">{waveform_spans()}</div>
            </div>
        </div>
        """
    )


def render_analysis_steps(*, has_file: bool = False, has_results: bool = False) -> None:
    step1 = "done" if has_file else "active"
    step2 = "done" if has_results else ("active" if has_file else "")
    step3 = "done" if has_results else ""
    render_html(
        f"""
        <div class="steps-section">
            <div class="steps-title">ขั้นตอนการวิเคราะห์</div>
            <div class="steps">
                <div class="step {step1}">
                    <div class="step-icon">📤</div>
                    <div class="step-num">01</div>
                    <div class="step-title">เตรียมไฟล์เสียง</div>
                    <div class="step-desc">อัปโหลดไฟล์เสียงหรือวิดีโอ MP4</div>
                </div>
                <div class="step {step2}">
                    <div class="step-icon">🔍</div>
                    <div class="step-num">02</div>
                    <div class="step-title">วิเคราะห์เสียง</div>
                    <div class="step-desc">กดปุ่มวิเคราะห์ ระบบประมวลผลทุกช่วง</div>
                </div>
                <div class="step {step3}">
                    <div class="step-icon">📈</div>
                    <div class="step-num">03</div>
                    <div class="step-title">แสดงผลลัพธ์</div>
                    <div class="step-desc">ดูคำตอบหลัก กราฟรายช่วง และดาวน์โหลดผลเป็น CSV</div>
                </div>
            </div>
        </div>
        """
    )


def render_page_home(bundles: list[ModelBundle]) -> None:
    render_hero(len(bundles))

    with st.container(border=True):
        render_upload_header()
        upload_col, preview_col = st.columns([1, 1], gap="large")

        with upload_col:
            uploaded = st.file_uploader(
                "เลือก WAV, MP3, FLAC, M4A หรือ MP4",
                type=sorted(
                    extension.removeprefix(".") for extension in SUPPORTED_EXTENSIONS
                ),
                help=(
                    f"ขนาดสูงสุด {MAX_FILE_SIZE_MB} MB "
                    f"และวิเคราะห์เฉพาะ {MAX_ANALYSIS_SEC} วินาทีแรก "
                    "ไฟล์ MP4 จะถูกดึงแทร็กเสียงมาวิเคราะห์อัตโนมัติ"
                ),
                label_visibility="collapsed",
                key="home_uploader",
            )

        with preview_col:
            if uploaded is None:
                render_file_preview_placeholder()
            else:
                file_size_mb = uploaded.size / (1024 * 1024)
                render_file_preview(uploaded.name, file_size_mb, len(bundles))

        result_key = None
        if uploaded is not None:
            remember_uploaded_file(uploaded)
            file_size_mb = uploaded.size / (1024 * 1024)
            result_key = (
                uploaded.name,
                uploaded.size,
                tuple(
                    (bundle.path.name, bundle.path.stat().st_mtime_ns)
                    for bundle in bundles
                ),
            )
            uploaded_suffix = Path(uploaded.name).suffix.lower()
            if uploaded_suffix in VIDEO_EXTENSIONS:
                st.video(uploaded)
                st.caption("ไฟล์วิดีโอ: ระบบจะดึงแทร็กเสียงมาวิเคราะห์")
            else:
                st.audio(uploaded)
            st.caption(
                f"ระบบวิเคราะห์เฉพาะ {MAX_ANALYSIS_SEC} วินาทีแรก "
                "ส่วนที่เกินจะไม่ถูกนำมาประมวลผล"
            )
            spectrogram_sr = bundles[0].config.sample_rate if bundles else 16000
            render_spectrogram(uploaded, sample_rate=spectrogram_sr)

            if st.button("🔍 เริ่มวิเคราะห์เสียง", type="primary", width="stretch"):
                if file_size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"ไฟล์ใหญ่เกิน {MAX_FILE_SIZE_MB} MB")
                else:
                    try:
                        with st.spinner(
                            f"กำลังวิเคราะห์เสียงด้วยโมเดลทั้ง {len(bundles)} ตัว..."
                        ):
                            results = analyze_upload(uploaded, bundles)
                            st.session_state.analysis_results = results
                            st.session_state.analysis_key = result_key
                            save_analysis_history(uploaded, results, bundles)
                    except Exception as exc:
                        st.error(f"วิเคราะห์ไม่สำเร็จ: {exc}")

    has_results = (
        uploaded is not None
        and result_key is not None
        and st.session_state.get("analysis_key") == result_key
    )
    render_analysis_steps(
        has_file=uploaded is not None,
        has_results=has_results,
    )

    if uploaded is None or not has_results:
        return

    results = st.session_state.analysis_results
    render_comparison(results, bundles)
    tabs = st.tabs([format_model_short(bundle.path) for bundle in bundles])
    for tab, bundle in zip(tabs, bundles):
        with tab:
            render_result(
                results[bundle.path.name],
                uploaded.name,
                format_model_name(bundle.path),
                bundle.path.name,
            )


def summarize_history_models(models: list[dict]) -> tuple[str, str, str]:
    """Return (summary_label, confidence_text, detail_text)."""
    if not models:
        return "—", "—", "—"

    labels = [str(item.get("label", "")).strip().lower() for item in models]
    fake_count = sum(1 for label in labels if label == "fake")
    real_count = sum(1 for label in labels if label == "real")

    if fake_count and not real_count:
        summary = "เสียงปลอม"
    elif real_count and not fake_count:
        summary = "เสียงจริง"
    elif fake_count or real_count:
        summary = "ผลไม่ตรงกัน"
    else:
        summary = "—"

    confidences = [
        float(item["confidence_pct"])
        for item in models
        if item.get("confidence_pct") is not None
    ]
    if confidences:
        confidence_text = f"{sum(confidences) / len(confidences):.1f}%"
    else:
        confidence_text = "—"

    detail_parts = []
    for item in models:
        short_name = MODEL_SHORT_NAMES.get(
            str(item.get("model_file", "")),
            str(item.get("model_file") or item.get("model_name") or "model"),
        )
        label = str(item.get("label", "—")).upper()
        conf = item.get("confidence_pct")
        conf_text = f"{float(conf):.1f}%" if conf is not None else "—"
        detail_parts.append(f"{short_name}: {label} ({conf_text})")

    return summary, confidence_text, " · ".join(detail_parts)


def render_page_history() -> None:
    render_html(
        """
        <div class="hero">
            <div class="hero-top">
                <div>
                    <h2>ประวัติการวิเคราะห์</h2>
                    <p>สรุปว่าไฟล์ที่วิเคราะห์เป็นเสียงจริงหรือเสียงปลอม และดาวน์โหลด CSV ได้ทันที</p>
                </div>
            </div>
        </div>
        """
    )
    entries = load_history()
    if not entries:
        st.info("ยังไม่มีประวัติการวิเคราะห์ ลองอัปโหลดไฟล์ที่หน้าแรกแล้วกดวิเคราะห์")
        return

    rows = []
    for entry in entries:
        models = entry.get("models") or []
        summary, confidence_text, detail = summarize_history_models(models)
        rows.append(
            {
                "เวลา": entry.get("timestamp", "—"),
                "ไฟล์": entry.get("file_name", "—"),
                "ขนาด (MB)": entry.get("file_size_mb", "—"),
                "ผลการวิเคราะห์": summary,
                "ความมั่นใจ": confidence_text,
                "รายละเอียดโมเดล": detail,
                "id": entry.get("id", ""),
            }
        )

    table = pd.DataFrame(rows)
    display_table = table.drop(columns=["id", "รายละเอียดโมเดล"])
    st.dataframe(display_table, width="stretch", hide_index=True)

    with st.expander("ดูรายละเอียดแยกตามโมเดล"):
        st.dataframe(
            table.drop(columns=["id"]),
            width="stretch",
            hide_index=True,
        )

    st.download_button(
        "💾 ดาวน์โหลดประวัติทั้งหมด (CSV)",
        data=table.drop(columns=["id"]).to_csv(index=False).encode("utf-8-sig"),
        file_name="analysis_history.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown("### จัดการรายการ")
    options = {
        f"{item.get('timestamp', '—')} · {item.get('file_name', '—')}": item.get("id")
        for item in entries
        if item.get("id")
    }
    selected = st.selectbox("เลือกรายการที่จะลบ", list(options.keys()))
    col_del, col_clear = st.columns(2)
    with col_del:
        if st.button("ลบรายการที่เลือก", width="stretch"):
            delete_history_entry(options[selected])
            st.success("ลบรายการแล้ว")
            st.rerun()
    with col_clear:
        if st.button("ล้างประวัติทั้งหมด", width="stretch", type="primary"):
            clear_history()
            st.success("ล้างประวัติแล้ว")
            st.rerun()


def render_page_files() -> None:
    render_html(
        """
        <div class="hero">
            <div class="hero-top">
                <div>
                    <h2>จัดการไฟล์เสียง</h2>
                    <p>รวมไฟล์ที่อัปโหลดในเซสชันนี้ และไฟล์ที่มีในประวัติการวิเคราะห์</p>
                </div>
            </div>
        </div>
        """
    )

    managed = dict(st.session_state.managed_files)
    for entry in load_history():
        name = entry.get("file_name")
        if not name:
            continue
        managed.setdefault(
            name,
            {
                "name": name,
                "size_mb": entry.get("file_size_mb", "—"),
                "suffix": Path(name).suffix.lower() or "—",
                "updated_at": entry.get("timestamp", "—"),
            },
        )

    if not managed:
        st.info("ยังไม่มีไฟล์ ลองอัปโหลดที่หน้าแรกหรือหน้า Speaker Diarization")
        return

    rows = pd.DataFrame(
        [
            {
                "ชื่อไฟล์": item["name"],
                "นามสกุล": item.get("suffix", "—"),
                "ขนาด (MB)": item.get("size_mb", "—"),
                "อัปเดตล่าสุด": item.get("updated_at", "—"),
            }
            for item in managed.values()
        ]
    )
    st.dataframe(rows, width="stretch", hide_index=True)

    selected_file = st.selectbox("เลือกไฟล์เพื่อจัดการ", list(managed.keys()))
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("ลบออกจากเซสชันนี้", width="stretch"):
            st.session_state.managed_files.pop(selected_file, None)
            st.success(f"ลบ `{selected_file}` ออกจากเซสชันแล้ว")
            st.rerun()
    with col_b:
        if st.button("ลบประวัติของไฟล์นี้", width="stretch", type="primary"):
            removed = delete_history_by_filename(selected_file)
            st.session_state.managed_files.pop(selected_file, None)
            st.success(f"ลบประวัติที่เกี่ยวข้อง {removed} รายการแล้ว")
            st.rerun()

    uploaded = st.file_uploader(
        "เพิ่มไฟล์เข้าชุดจัดการ",
        type=sorted(extension.removeprefix(".") for extension in SUPPORTED_EXTENSIONS),
        key="files_uploader",
    )
    if uploaded is not None:
        is_new = uploaded.name not in st.session_state.managed_files
        remember_uploaded_file(uploaded)
        if is_new:
            st.success(f"เพิ่ม `{uploaded.name}` แล้ว")
            st.rerun()
        if Path(uploaded.name).suffix.lower() in VIDEO_EXTENSIONS:
            st.video(uploaded)
        else:
            st.audio(uploaded)


def render_page_diarization() -> None:
    render_html(
        """
        <div class="hero">
            <div class="hero-top">
                <div>
                    <h2>Speaker Diarization</h2>
                    <p>แยกช่วงที่มีเสียงพูดด้วยพลังงานของสัญญาณ (Speech Activity Detection)
                    ใช้งานได้บนเครื่องโดยไม่ต้องพึ่งโมเดลภายนอก</p>
                </div>
            </div>
        </div>
        """
    )
    st.caption(
        "หมายเหตุ: โหมดนี้แยกช่วงที่มี/ไม่มีเสียงพูด ไม่ได้ระบุตัวตนผู้พูดแบบ multi-speaker ID"
    )

    uploaded = st.file_uploader(
        "อัปโหลดไฟล์สำหรับแยกช่วงเสียงพูด",
        type=sorted(extension.removeprefix(".") for extension in SUPPORTED_EXTENSIONS),
        key="diarization_uploader",
    )
    if uploaded is None:
        st.info("อัปโหลดไฟล์เสียงหรือวิดีโอ MP4 เพื่อเริ่มแยกช่วง")
        return

    remember_uploaded_file(uploaded)
    if Path(uploaded.name).suffix.lower() in VIDEO_EXTENSIONS:
        st.video(uploaded)
    else:
        st.audio(uploaded)

    if not st.button("🔎 เริ่มแยกช่วงเสียงพูด", type="primary", width="stretch"):
        return

    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(uploaded.getvalue())
        temp_path = Path(temp_file.name)
    try:
        with st.spinner("กำลังวิเคราะห์ช่วงเสียงพูด..."):
            audio = limit_audio_duration(
                load_audio_file(
                    temp_path,
                    DIARIZATION_SR,
                    max_duration_sec=MAX_ANALYSIS_SEC,
                ),
                DIARIZATION_SR,
                MAX_ANALYSIS_SEC,
            )
            segments = detect_speech_segments(audio, DIARIZATION_SR)
    except Exception as exc:
        st.error(f"แยกช่วงไม่สำเร็จ: {exc}")
        return
    finally:
        temp_path.unlink(missing_ok=True)

    if not segments:
        st.warning("ไม่พบช่วงที่น่าจะเป็นเสียงพูดในไฟล์นี้")
        return

    rows = pd.DataFrame(
        [
            {
                "ช่วง": segment.index,
                "เริ่ม (วิ)": segment.start_sec,
                "สิ้นสุด (วิ)": segment.end_sec,
                "ความยาว (วิ)": segment.duration_sec,
                "RMS": round(segment.rms, 6),
            }
            for segment in segments
        ]
    )
    st.success(f"พบช่วงเสียงพูด {len(segments)} ช่วง ใน {MAX_ANALYSIS_SEC} วินาทีแรก")
    st.dataframe(rows, width="stretch", hide_index=True)

    chart = (
        alt.Chart(rows)
        .mark_bar(cornerRadius=6, color="#3b82f6")
        .encode(
            x=alt.X("เริ่ม (วิ):Q", title="เวลาเริ่ม (วินาที)"),
            x2="สิ้นสุด (วิ):Q",
            y=alt.Y("ช่วง:O", title="ช่วง"),
            tooltip=["ช่วง", "เริ่ม (วิ)", "สิ้นสุด (วิ)", "ความยาว (วิ)", "RMS"],
        )
        .properties(height=220)
        .configure(background="transparent")
    )
    st.altair_chart(chart, width="stretch")
    st.download_button(
        "💾 บันทึกช่วงเสียงพูด (CSV)",
        data=rows.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{Path(uploaded.name).stem}_speech_segments.csv",
        mime="text/csv",
        width="stretch",
    )


def render_page_settings(bundles: list[ModelBundle]) -> None:
    render_html(
        """
        <div class="hero">
            <div class="hero-top">
                <div>
                    <h2>ตั้งค่า</h2>
                    <p>ดูค่าคอนฟิกของระบบ ล้างแคชโมเดล และจัดการข้อมูลประวัติ</p>
                </div>
            </div>
        </div>
        """
    )

    st.markdown("### ค่าการวิเคราะห์")
    st.write(f"- ขนาดไฟล์สูงสุด: **{MAX_FILE_SIZE_MB} MB**")
    st.write(f"- วิเคราะห์เฉพาะ: **{MAX_ANALYSIS_SEC} วินาทีแรก**")
    st.write(f"- โฟลเดอร์โมเดล: `{MODELS_DIR}`")
    st.write(f"- โฟลเดอร์ประวัติ: `{DATA_DIR}`")
    st.write(
        f"- รูปแบบที่รองรับ: "
        f"**{', '.join(sorted(ext.lstrip('.') for ext in SUPPORTED_EXTENSIONS))}**"
    )

    st.markdown("### โมเดลที่โหลดอยู่")
    if not bundles:
        st.warning("ยังไม่พบโมเดล")
    else:
        for bundle in bundles:
            st.write(
                f"- `{bundle.path.name}` · device `{bundle.device}` · "
                f"LFCC {bundle.config.n_lfcc}×{bundle.config.frame_count}"
            )

    st.markdown("### การจัดการระบบ")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("รีเฟรชแคชโมเดล", width="stretch"):
            clear_model_cache()
            st.success("ล้างแคชโมเดลแล้ว")
            st.rerun()
    with col_b:
        if st.button("ล้างประวัติทั้งหมด", width="stretch"):
            clear_history()
            st.success("ล้างประวัติแล้ว")
    with col_c:
        if st.button("ล้างไฟล์ในเซสชัน", width="stretch"):
            st.session_state.managed_files = {}
            st.success("ล้างรายการไฟล์ในเซสชันแล้ว")

    st.markdown("### ไปยังหน้าอื่นอย่างรวดเร็ว")
    quick_cols = st.columns(4)
    targets = [
        ("home", "หน้าแรก"),
        ("history", "ประวัติ"),
        ("files", "ไฟล์เสียง"),
        ("diarization", "Diarization"),
    ]
    for column, (key, label) in zip(quick_cols, targets):
        with column:
            if st.button(label, width="stretch", key=f"quick_{key}"):
                st.session_state.page = key
                st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Voice Authenticity Detector",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()
    init_session_state()
    render_sidebar_brand()
    page = render_sidebar_nav()

    bundles: list[ModelBundle] = []
    if page in {"home", "settings"}:
        bundles = load_models(required=(page == "home"))
        render_sidebar_models(bundles)
    else:
        with st.sidebar:
            render_html(
                f"""
                <div class="side-footer">
                    หน้าปัจจุบัน: {NAV_PAGES.get(page, page)}<br>
                    รองรับ WAV · MP3 · FLAC · M4A · MP4<br>
                    © 2026 Voice Authenticity Detector
                </div>
                """
            )

    if page == "home":
        render_page_home(bundles)
    elif page == "history":
        render_page_history()
    elif page == "files":
        render_page_files()
    elif page == "diarization":
        render_page_diarization()
    elif page == "settings":
        if not bundles:
            bundles = load_models(required=False)
        render_page_settings(bundles)


if __name__ == "__main__":
    main()
