import streamlit as st
import streamlit.components.v1 as components
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
import io
import json
import struct
import wave
import base64
import time
from datetime import datetime

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

EMOTION_COLORS = {
    "neutral": "#808080",
    "happy": "#FFD700",
    "sad": "#4169E1",
    "angry": "#FF4500",
    "fear": "#800080",
    "surprise": "#FF69B4",
    "disgust": "#006400"
}

EMOTION_LABELS_CN = {
    "neutral": "中性",
    "happy": "高兴",
    "sad": "悲伤",
    "angry": "愤怒",
    "fear": "恐惧",
    "surprise": "惊讶",
    "disgust": "厌恶"
}


def call_analyze_api(file_bytes, filename, model_mode, is_dual_channel, output_format):
    url = f"{API_BASE}/api/analyze"
    files = {"file": (filename, file_bytes)}
    params = {"model_mode": model_mode, "is_dual_channel": is_dual_channel,
              "output_format": output_format}
    try:
        resp = requests.post(url, files=files, params=params, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API调用失败: {e}")
        return None


def call_health_api():
    url = f"{API_BASE}/api/health"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _parse_audio_bytes(file_bytes, filename):
    ext = os.path.splitext(filename or "audio.wav")[1].lower()
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception:
        pass
    try:
        import librosa
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(file_bytes)
        tmp.close()
        audio, sr = librosa.load(tmp.name, sr=None, mono=True)
        os.unlink(tmp.name)
        return audio, sr
    except Exception:
        pass
    return None, None


def _audio_segment_to_wav_bytes(audio_segment, sr):
    pcm = (audio_segment * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf.read()


def _build_annotation_data(result, filename, analyzed_at=None):
    audio_info = result.get("audio_info", {})
    sentences = result.get("sentences", [])
    summary = result.get("dialogue_summary", {})

    turning_points = summary.get("turning_points", [])
    escalation_intervals = summary.get("escalation_intervals", [])

    sentence_annotations = []
    for idx, sent in enumerate(sentences):
        is_turning_point = any(
            abs(sent["start_time"] - tp["time"]) < 0.01 for tp in turning_points
        )

        is_escalation = any(
            esc["start_time"] <= sent["start_time"] <= esc["end_time"]
            for esc in escalation_intervals
        )

        sentence_annotations.append({
            "sentence_id": idx + 1,
            "start_time": sent["start_time"],
            "end_time": sent["end_time"],
            "speaker_id": sent["speaker_id"],
            "emotion": sent["emotion"],
            "confidence": sent["confidence"],
            "valence": sent["valence"],
            "arousal": sent["arousal"],
            "is_turning_point": is_turning_point,
            "is_escalation": is_escalation
        })

    dialogue_summary = {
        "dominant_emotion": summary.get("dominant_emotion", "neutral"),
        "valence_std": summary.get("valence_std", 0.0),
        "conflict_density": summary.get("conflict_density", 0.0),
        "turning_point_count": len(turning_points),
        "escalation_interval_count": len(escalation_intervals),
        "contagion_event_count": len(summary.get("contagion_events", []))
    }

    if analyzed_at is None:
        analyzed_at = datetime.now().isoformat()

    annotation = {
        "audio_metadata": {
            "filename": filename or "unknown.wav",
            "duration": audio_info.get("duration", 0.0),
            "sample_rate": audio_info.get("sample_rate", 0),
            "channels": audio_info.get("channels", 0),
            "analyzed_at": analyzed_at
        },
        "sentences": sentence_annotations,
        "dialogue_summary": dialogue_summary
    }
    return annotation


def _generate_csv_annotation(annotation):
    sentences = annotation.get("sentences", [])
    if not sentences:
        return ""

    import csv
    output = io.StringIO()
    fieldnames = [
        "sentence_id", "start_time", "end_time", "speaker_id",
        "emotion", "confidence", "valence", "arousal",
        "is_turning_point", "is_escalation"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for sent in sentences:
        writer.writerow(sent)

    return output.getvalue()


def plot_waveform_with_emotions(result, audio_data=None, sr=None):
    sentences = result.get("sentences", [])
    if not sentences:
        st.info("没有检测到语音段")
        return

    fig = go.Figure()

    if audio_data is not None and sr is not None:
        duration = len(audio_data) / sr
        time_axis = np.linspace(0, duration, len(audio_data))
        downsample = max(1, len(audio_data) // 8000)
        t_ds = time_axis[::downsample]
        y_ds = audio_data[::downsample]
        fig.add_trace(go.Scatter(
            x=t_ds, y=y_ds, mode='lines',
            line=dict(color='silver', width=1),
            name='波形',
            hoverinfo='skip'
        ))
        for sent in sentences:
            emotion = sent["emotion"]
            color = EMOTION_COLORS.get(emotion, "#808080")
            mid = (sent["start_time"] + sent["end_time"]) / 2
            fig.add_vrect(
                x0=sent["start_time"], x1=sent["end_time"],
                fillcolor=color, opacity=0.25,
                line_width=0
            )
            label = EMOTION_LABELS_CN.get(emotion, emotion)
            fig.add_annotation(
                x=mid, y=np.max(np.abs(y_ds)) * 0.85,
                text=label, showarrow=False,
                font=dict(color=color, size=11, family="Arial Black")
            )
        y_max = np.max(np.abs(y_ds)) * 1.2 if len(y_ds) > 0 else 1
        fig.update_yaxes(range=[-y_max, y_max])
    else:
        for i, sent in enumerate(sentences):
            emotion = sent["emotion"]
            color = EMOTION_COLORS.get(emotion, "#808080")
            label = f'{EMOTION_LABELS_CN.get(emotion, emotion)} ({sent["confidence"]:.2f})'
            fig.add_trace(go.Scatter(
                x=[sent["start_time"], sent["end_time"]],
                y=[i % 2, i % 2],
                mode='lines+markers',
                line=dict(color=color, width=8),
                marker=dict(size=10),
                name=f'句{i+1}: {label}',
                hovertext=f'{sent["speaker_id"]} | {sent["start_time"]:.1f}s-{sent["end_time"]:.1f}s<br>'
                          f'情感: {label}<br>效价: {sent["valence"]:.2f} | 唤醒度: {sent["arousal"]:.2f}',
                hoverinfo='text'
            ))

    fig.update_layout(
        title="语音波形与情感标注",
        xaxis_title="时间 (秒)",
        yaxis_title="幅度",
        showlegend=True,
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_emotion_trajectory(result, highlight_idx=None):
    sentences = result.get("sentences", [])
    summary = result.get("dialogue_summary", {})
    if not sentences:
        return

    times = [(s["start_time"] + s["end_time"]) / 2 for s in sentences]
    valences = [s["valence"] for s in sentences]
    arousals = [s["arousal"] for s in sentences]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("效价 (Valence) 轨迹", "唤醒度 (Arousal) 轨迹"))

    marker_sizes_valence = [8] * len(sentences)
    marker_sizes_arousal = [8] * len(sentences)
    marker_colors_valence = ["blue"] * len(sentences)
    marker_colors_arousal = ["red"] * len(sentences)

    if highlight_idx is not None and 0 <= highlight_idx < len(sentences):
        marker_sizes_valence[highlight_idx] = 18
        marker_sizes_arousal[highlight_idx] = 18
        marker_colors_valence[highlight_idx] = "gold"
        marker_colors_arousal[highlight_idx] = "gold"

    fig.add_trace(go.Scatter(
        x=times, y=valences, mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=marker_sizes_valence, color=marker_colors_valence,
                    line=dict(color='darkblue', width=1)),
        name='效价'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=times, y=arousals, mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=marker_sizes_arousal, color=marker_colors_arousal,
                    line=dict(color='darkred', width=1)),
        name='唤醒度'
    ), row=2, col=1)

    for tp in summary.get("turning_points", []):
        fig.add_vline(x=tp["time"], line_dash="dash", line_color="orange",
                      annotation_text="转折", row=1, col=1)

    for esc in summary.get("escalation_intervals", []):
        fig.add_vrect(x0=esc["start_time"], x1=esc["end_time"],
                      fillcolor="red", opacity=0.1,
                      annotation_text="激化", row=2, col=1)

    fig.update_layout(height=500, title_text="情绪轨迹曲线", showlegend=False)
    fig.update_yaxes(range=[-1.1, 1.1], row=1, col=1)
    fig.update_yaxes(range=[-1.1, 1.1], row=2, col=1)
    fig.update_xaxes(title_text="时间 (秒)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def plot_emotion_pie(result):
    sentences = result.get("sentences", [])
    if not sentences:
        return

    from collections import Counter
    counts = Counter([s["emotion"] for s in sentences])
    labels = [EMOTION_LABELS_CN.get(k, k) for k in counts.keys()]
    colors = [EMOTION_COLORS.get(k, "#808080") for k in counts.keys()]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=list(counts.values()),
        marker=dict(colors=colors),
        textinfo='label+percent',
        hole=0.3
    )])
    fig.update_layout(title="情感分布", height=400)
    st.plotly_chart(fig, use_container_width=True)


def plot_dual_speaker_trajectory(result):
    sentences = result.get("sentences", [])
    summary = result.get("dialogue_summary", [])
    if not sentences:
        return

    speakers = sorted(set([s["speaker_id"] for s in sentences]))
    if len(speakers) < 2:
        st.info("非双人对话模式，无传染分析")
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f"{speakers[0]} 情绪轨迹", f"{speakers[1]} 情绪轨迹"))

    speaker_colors = {speakers[0]: "blue", speakers[1]: "green"}

    for idx, speaker in enumerate(speakers):
        spk_sents = [s for s in sentences if s["speaker_id"] == speaker]
        times = [(s["start_time"] + s["end_time"]) / 2 for s in spk_sents]
        valences = [s["valence"] for s in spk_sents]
        fig.add_trace(go.Scatter(
            x=times, y=valences, mode='lines+markers',
            line=dict(color=speaker_colors[speaker], width=2),
            marker=dict(size=8),
            name=f'{speaker} 效价'
        ), row=idx + 1, col=1)

    contagion_events = summary.get("contagion_events", [])
    for ce in contagion_events:
        fig.add_annotation(
            x=ce["target_time"], y=-0.8,
            text=f"← 传染 (延迟{ce['delay_sentences']}句)",
            showarrow=True, arrowhead=2,
            ax=ce["source_time"] - ce["target_time"],
            ay=0,
            row=2 if ce["target_speaker"] == speakers[1] else 1, col=1
        )

    fig.update_layout(height=600, title_text="双人对话情绪曲线与传染关联")
    fig.update_yaxes(range=[-1.1, 1.1])
    st.plotly_chart(fig, use_container_width=True)


def render_sentence_table(result):
    sentences = result.get("sentences", [])
    if not sentences:
        return

    df_data = []
    for i, s in enumerate(sentences):
        df_data.append({
            "句号": i + 1,
            "说话人": s["speaker_id"],
            "起止时间": f'{s["start_time"]:.1f}s - {s["end_time"]:.1f}s',
            "情感": EMOTION_LABELS_CN.get(s["emotion"], s["emotion"]),
            "置信度": f'{s["confidence"]:.3f}',
            "效价": f'{s["valence"]:.3f}',
            "唤醒度": f'{s["arousal"]:.3f}'
        })
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)


def render_export_buttons(result, filename, analyzed_at=None):
    annotation = _build_annotation_data(result, filename, analyzed_at)

    col1, col2 = st.columns(2)
    with col1:
        json_str = json.dumps(annotation, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 导出 JSON 标注",
            data=json_str,
            file_name=f"annotation_{result.get('task_id', 'result')}.json",
            mime="application/json",
            use_container_width=True
        )
    with col2:
        csv_str = _generate_csv_annotation(annotation)
        st.download_button(
            label="📊 导出 CSV 标注",
            data=csv_str,
            file_name=f"annotation_{result.get('task_id', 'result')}.csv",
            mime="text/csv",
            use_container_width=True
        )


def render_dialogue_playback(result, audio_raw, audio_sr, is_dual):
    sentences = result.get("sentences", [])
    if not sentences or audio_raw is None or audio_sr is None:
        st.info("音频数据不可用，无法回放")
        return

    num_sentences = len(sentences)

    if "playback_current" not in st.session_state:
        st.session_state["playback_current"] = 0
    if "playback_playing" not in st.session_state:
        st.session_state["playback_playing"] = False

    query_params = st.experimental_get_query_params()
    if "pb_adv" in query_params and st.session_state["playback_playing"]:
        adv = int(query_params["pb_adv"][0])
        if adv == 1:
            if st.session_state["playback_current"] < num_sentences - 1:
                st.session_state["playback_current"] += 1
            else:
                st.session_state["playback_playing"] = False
            st.experimental_set_query_params()

    current_idx = st.session_state["playback_current"]
    is_playing = st.session_state["playback_playing"]

    st.subheader("🎬 对话回放")

    progress_value = (current_idx + 1) / num_sentences if num_sentences > 0 else 0
    st.progress(progress_value)
    st.caption(f"播放进度: {current_idx + 1}/{num_sentences}")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        if st.button("▶️ 播放", use_container_width=True, key="play_btn"):
            st.session_state["playback_playing"] = True
            if st.session_state["playback_current"] >= num_sentences:
                st.session_state["playback_current"] = 0
            st.rerun()
    with col2:
        if st.button("⏸️ 暂停", use_container_width=True, key="pause_btn"):
            st.session_state["playback_playing"] = False
            st.experimental_set_query_params()
            st.rerun()
    with col3:
        if st.button("⏮️ 重置", use_container_width=True, key="reset_btn"):
            st.session_state["playback_playing"] = False
            st.session_state["playback_current"] = 0
            st.experimental_set_query_params()
            st.rerun()
    with col4:
        jump_idx = st.number_input(
            "跳转到句子",
            min_value=1,
            max_value=num_sentences,
            value=current_idx + 1,
            step=1,
            key="jump_input"
        )
        if st.button("跳转", use_container_width=True, key="jump_btn"):
            st.session_state["playback_current"] = jump_idx - 1
            st.session_state["playback_playing"] = False
            st.experimental_set_query_params()
            st.rerun()

    if 0 <= current_idx < num_sentences:
        current_sent = sentences[current_idx]
        emotion = current_sent["emotion"]
        color = EMOTION_COLORS.get(emotion, "#808080")
        speaker_id = current_sent["speaker_id"]

        speaker_bg = "#E8F4FD" if speaker_id == "speaker_0" else "#F0FFF0"
        if not is_dual:
            speaker_bg = "#F5F5F5"

        st.markdown(
            f'<div style="padding: 15px; border-radius: 10px; background-color: {speaker_bg}; '
            f'border-left: 5px solid {color}; margin: 10px 0;">'
            f'<strong>句{current_idx + 1}</strong> | {speaker_id} | '
            f'{current_sent["start_time"]:.1f}s - {current_sent["end_time"]:.1f}s<br>'
            f'<span style="color: {color}; font-weight: bold;">'
            f'{EMOTION_LABELS_CN.get(emotion, emotion)}</span> '
            f'(置信度: {current_sent["confidence"]:.2f})<br>'
            f'效价: {current_sent["valence"]:.2f} | 唤醒度: {current_sent["arousal"]:.2f}'
            f'</div>',
            unsafe_allow_html=True
        )

        start_sample = int(current_sent["start_time"] * audio_sr)
        end_sample = int(current_sent["end_time"] * audio_sr)
        end_sample = min(end_sample, len(audio_raw))
        if start_sample < end_sample:
            segment = audio_raw[start_sample:end_sample]
            wav_bytes = _audio_segment_to_wav_bytes(segment, audio_sr)
            audio_key = f"audio_pb_{current_idx}_{'p' if is_playing else 's'}"
            st.audio(wav_bytes, format="audio/wav", autoplay=is_playing, key=audio_key)

        with st.expander("📈 当前情绪轨迹位置", expanded=True):
            summary = result.get("dialogue_summary", {})
            times = [(s["start_time"] + s["end_time"]) / 2 for s in sentences]
            valences = [s["valence"] for s in sentences]
            arousals = [s["arousal"] for s in sentences]

            marker_sizes_v = [6] * len(sentences)
            marker_sizes_a = [6] * len(sentences)
            marker_colors_v = ["blue"] * len(sentences)
            marker_colors_a = ["red"] * len(sentences)

            if 0 <= current_idx < len(sentences):
                marker_sizes_v[current_idx] = 16
                marker_sizes_a[current_idx] = 16
                marker_colors_v[current_idx] = "gold"
                marker_colors_a[current_idx] = "gold"

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("效价轨迹", "唤醒度轨迹"),
                                vertical_spacing=0.08)

            fig.add_trace(go.Scatter(
                x=times, y=valences, mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=marker_sizes_v, color=marker_colors_v,
                            line=dict(color='darkblue', width=1)),
                name='效价'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=times, y=arousals, mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=marker_sizes_a, color=marker_colors_a,
                            line=dict(color='darkred', width=1)),
                name='唤醒度'
            ), row=2, col=1)

            for tp in summary.get("turning_points", []):
                fig.add_vline(x=tp["time"], line_dash="dash", line_color="orange",
                              line_width=1, row=1, col=1)

            for esc in summary.get("escalation_intervals", []):
                fig.add_vrect(x0=esc["start_time"], x1=esc["end_time"],
                              fillcolor="red", opacity=0.1,
                              line_width=0, row=2, col=1)

            fig.update_layout(height=280, showlegend=False, margin=dict(t=30, b=30))
            fig.update_yaxes(range=[-1.1, 1.1], row=1, col=1)
            fig.update_yaxes(range=[-1.1, 1.1], row=2, col=1)
            fig.update_xaxes(title_text="时间 (秒)", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.caption("句子列表（当前播放句高亮显示）")

    for i, sent in enumerate(sentences):
        is_current = (i == current_idx)
        sent_emotion = sent["emotion"]
        sent_color = EMOTION_COLORS.get(sent_emotion, "#808080")
        sent_speaker = sent["speaker_id"]

        if is_current:
            bg_color = "#FFFACD" if not is_dual else ("#E8F4FD" if sent_speaker == "speaker_0" else "#F0FFF0")
            border_style = f"2px solid {sent_color}"
        else:
            bg_color = "#FAFAFA"
            border_style = "1px solid #E0E0E0"

        speaker_label = f"[{sent_speaker}] " if is_dual else ""
        st.markdown(
            f'<div style="padding: 8px; margin: 4px 0; border-radius: 5px; '
            f'background-color: {bg_color}; border: {border_style};">'
            f'<strong>句{i + 1}</strong> {speaker_label}'
            f'<span style="color: {sent_color};">● {EMOTION_LABELS_CN.get(sent_emotion, sent_emotion)}</span> '
            f'| {sent["start_time"]:.1f}s - {sent["end_time"]:.1f}s'
            f'</div>',
            unsafe_allow_html=True
        )

    if is_playing and 0 <= current_idx < num_sentences:
        sent_duration = sentences[current_idx]["end_time"] - sentences[current_idx]["start_time"]
        pause_duration = 0.5
        wait_secs = sent_duration + pause_duration

        auto_advance_js = f"""
        <script>
        (function() {{
            const waitMs = {int(wait_secs * 1000)};
            setTimeout(function() {{
                const currentUrl = new URL(window.parent.location.href);
                currentUrl.searchParams.set('pb_adv', '1');
                window.parent.location.href = currentUrl.toString();
            }}, waitMs);
        }})();
        </script>
        """
        components.html(auto_advance_js, height=0, width=0)


def render_summary(result):
    summary = result.get("dialogue_summary", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        dominant = summary.get("dominant_emotion", "neutral")
        st.metric("主导情感", EMOTION_LABELS_CN.get(dominant, dominant))
    with col2:
        st.metric("情绪波动度", f'{summary.get("valence_std", 0):.3f}')
    with col3:
        st.metric("冲突密度", f'{summary.get("conflict_density", 0):.4f}')

    tps = summary.get("turning_points", [])
    if tps:
        st.subheader("情绪转折点")
        for tp in tps:
            st.markdown(f"- **{tp['time']:.1f}s**: 效价从 {tp['valence_before']:.2f} → {tp['valence_after']:.2f} (Δ={tp['delta']:.2f})")

    escs = summary.get("escalation_intervals", [])
    if escs:
        st.subheader("情绪激化区间")
        for esc in escs:
            st.markdown(f"- **{esc['start_time']:.1f}s - {esc['end_time']:.1f}s**: 平均唤醒度 {esc['avg_arousal']:.2f}")

    conts = summary.get("contagion_events", [])
    if conts:
        st.subheader("情绪传染事件")
        for ce in conts:
            st.markdown(f"- {ce['source_speaker']} ({ce['source_time']:.1f}s) → {ce['target_speaker']} ({ce['target_time']:.1f}s): "
                       f"延迟{ce['delay_sentences']}句, 强度{ce['contagion_strength']:.2f}")


def call_batch_submit_api(files, batch_config):
    url = f"{API_BASE}/api/batch/submit"
    files_data = [("files", (f.name, f.getvalue())) for f in files]
    params = {"batch_config": json.dumps(batch_config)}
    try:
        resp = requests.post(url, files=files_data, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"批量提交失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                detail = e.response.json().get("detail", e.response.text)
                st.error(f"错误详情: {detail}")
            except Exception:
                st.error(f"响应: {e.response.text}")
        return None


def call_batch_status_api(batch_id):
    url = f"{API_BASE}/api/batch/{batch_id}/status"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return None


def call_batch_report_api(batch_id, format="json"):
    url = f"{API_BASE}/api/batch/{batch_id}/report"
    params = {"format": format}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 202:
            return {"status": "processing", "data": resp.json()}
        resp.raise_for_status()
        if format == "json":
            return {"status": "completed", "data": resp.json()}
        else:
            return {"status": "completed", "data": resp.text}
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
            return {"status": "not_found", "data": None}
        return {"status": "error", "data": str(e)}


def plot_batch_emotion_radar(report):
    comparison = report.get("comparison", {})
    if "emotion_distribution" not in comparison:
        return

    dist = comparison["emotion_distribution"]["distributions"]
    file_summaries = report.get("file_summaries", {})

    fig = go.Figure()

    for fid, emotions in dist.items():
        fname = file_summaries.get(fid, {}).get("filename", fid)
        values = [emotions[e] for e in EMOTION_LABELS]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[EMOTION_LABELS_CN.get(e, e) for e in EMOTION_LABELS],
            fill='toself',
            name=fname,
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="情感分布雷达图对比",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_batch_emotion_bar(report):
    comparison = report.get("comparison", {})
    if "emotion_distribution" not in comparison:
        return

    dist = comparison["emotion_distribution"]["distributions"]
    file_summaries = report.get("file_summaries", {})

    fig = go.Figure()
    x_labels = [EMOTION_LABELS_CN.get(e, e) for e in EMOTION_LABELS]

    for fid, emotions in dist.items():
        fname = file_summaries.get(fid, {}).get("filename", fid)
        fig.add_trace(go.Bar(
            name=fname,
            x=x_labels,
            y=[emotions[e] for e in EMOTION_LABELS]
        ))

    fig.update_layout(
        barmode='group',
        title="情感分布柱状图对比",
        xaxis_title="情感类型",
        yaxis_title="占比",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_valence_trend_comparison(report):
    comparison = report.get("comparison", {})
    if "valence_trend" not in comparison:
        return

    trends = comparison["valence_trend"]["trends"]
    file_summaries = report.get("file_summaries", {})

    fig = go.Figure()

    for fid, data in trends.items():
        fname = file_summaries.get(fid, {}).get("filename", fid)
        valences = data["valence_values"]
        x = list(range(len(valences)))
        fig.add_trace(go.Scatter(
            x=x, y=valences, mode='lines',
            name=f'{fname} ({data["trend"]}, slope={data["slope"]:.4f})',
            line=dict(width=2)
        ))

    fig.update_layout(
        title="效价趋势曲线对比",
        xaxis_title="句子序号",
        yaxis_title="效价 (Valence)",
        yaxis_range=[-1.1, 1.1],
        height=450,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_arousal_pattern_comparison(report):
    comparison = report.get("comparison", {})
    if "arousal_pattern" not in comparison:
        return

    patterns = comparison["arousal_pattern"]["patterns"]
    file_summaries = report.get("file_summaries", {})

    fids = list(patterns.keys())
    fnames = [file_summaries.get(f, {}).get("filename", f) for f in fids]
    stds = [patterns[f]["arousal_std"] for f in fids]
    means = [patterns[f]["mean_arousal"] for f in fids]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fnames, y=stds, name="唤醒度标准差",
        marker_color='indianred', opacity=0.7
    ))
    fig.add_trace(go.Bar(
        x=fnames, y=means, name="平均唤醒度",
        marker_color='lightsalmon', opacity=0.7
    ))

    fig.update_layout(
        barmode='group',
        title="唤醒度模式对比",
        xaxis_title="文件",
        yaxis_title="数值",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_baseline_deviation_heatmap(report):
    comparison = report.get("comparison", {})
    meta = report.get("meta", {})
    baseline_file = meta.get("baseline_file")
    if not baseline_file:
        return

    file_summaries = report.get("file_summaries", {})
    baseline_name = file_summaries.get(baseline_file, {}).get("filename", "基线")

    deviation_data = []
    dimensions = []
    file_names = []

    if "emotion_distribution" in comparison and "baseline_deviations" in comparison["emotion_distribution"]:
        devs = comparison["emotion_distribution"]["baseline_deviations"]
        for fid, dev in devs.items():
            if fid not in file_names:
                file_names.append(fid)
            deviation_data.append(abs(dev["js_divergence"]))
        dimensions.append("情感分布(JS散度)")

    if "valence_trend" in comparison and "baseline_deviations" in comparison["valence_trend"]:
        devs = comparison["valence_trend"]["baseline_deviations"]
        for fid, dev in devs.items():
            if fid not in file_names:
                file_names.append(fid)
            deviation_data.append(abs(dev["slope_deviation"]))
        dimensions.append("效价趋势(斜率差)")

    if "arousal_pattern" in comparison and "baseline_deviations" in comparison["arousal_pattern"]:
        devs = comparison["arousal_pattern"]["baseline_deviations"]
        for fid, dev in devs.items():
            if fid not in file_names:
                file_names.append(fid)
            deviation_data.append(abs(dev["std_deviation"]))
        dimensions.append("唤醒度模式(标准差差)")

    if not deviation_data or not dimensions:
        return

    n_files = len(file_names)
    n_dims = len(dimensions)
    heatmap_data = np.array(deviation_data).reshape(n_dims, n_files)

    display_names = [file_summaries.get(f, {}).get("filename", f) for f in file_names]

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=display_names,
        y=dimensions,
        colorscale='Reds',
        showscale=True,
        text=[[f"{v:.4f}" for v in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 12}
    ))

    fig.update_layout(
        title=f"基线偏差热力图 (基线: {baseline_name})",
        xaxis_title="对比文件",
        yaxis_title="对比维度",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_similarity_matrix(report):
    comparison = report.get("comparison", {})
    if "speaker_similarity" not in comparison:
        return

    sim_matrix = comparison["speaker_similarity"].get("cross_file_similarity_matrix", {})
    file_summaries = report.get("file_summaries", {})
    dual_files = comparison["speaker_similarity"].get("dual_channel_files", [])

    if len(dual_files) < 2:
        st.info("双通道文件不足2个，无法生成相似度矩阵")
        return

    display_names = [file_summaries.get(f, {}).get("filename", f) for f in dual_files]
    n = len(dual_files)
    matrix = np.ones((n, n))

    for key, value in sim_matrix.items():
        f1, f2 = key.split("__")
        if f1 in dual_files and f2 in dual_files:
            i = dual_files.index(f1)
            j = dual_files.index(f2)
            matrix[i][j] = value
            matrix[j][i] = value

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=display_names,
        y=display_names,
        colorscale='Blues',
        showscale=True,
        zmin=0, zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 12}
    ))

    fig.update_layout(
        title="双通道文件说话人同步性相似度矩阵",
        xaxis_title="文件",
        yaxis_title="文件",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)


def render_batch_page():
    st.header("📊 批量对比分析")

    health = call_health_api()
    if health:
        st.sidebar.success(f"API服务运行中 v{health.get('version', '?')}")
    else:
        st.sidebar.error("API服务未连接")

    st.subheader("1. 上传音频文件")
    uploaded_files = st.file_uploader(
        "选择多个音频文件 (2-20个, 支持WAV/MP3/FLAC/OGG/M4A)",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"已选择 {len(uploaded_files)} 个文件")
        for f in uploaded_files:
            st.caption(f"📄 {f.name} ({f.size/1024/1024:.2f} MB)")

    st.subheader("2. 对比配置")

    batch_name = st.text_input("批次名称", value=f"批次_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    dimension_options = [
        ("emotion_distribution", "情感分布对比"),
        ("valence_trend", "效价趋势对比"),
        ("arousal_pattern", "唤醒度模式对比"),
        ("speaker_similarity", "说话人同步性对比")
    ]

    selected_dims = st.multiselect(
        "选择对比维度 (至少选择1个)",
        [d[0] for d in dimension_options],
        format_func=lambda x: dict(dimension_options)[x],
        default=[d[0] for d in dimension_options]
    )

    baseline_file = None
    if uploaded_files:
        baseline_file = st.selectbox(
            "选择基线文件 (可选，其他文件将与此文件对比)",
            ["无"] + [f.name for f in uploaded_files]
        )
        if baseline_file == "无":
            baseline_file = None

    st.subheader("3. 提交任务")

    if st.button("🚀 提交批量分析任务", type="primary", disabled=not (uploaded_files and selected_dims)):
        if not uploaded_files or len(uploaded_files) < 2:
            st.error("请至少上传2个音频文件")
        elif not selected_dims:
            st.error("请至少选择1个对比维度")
        elif len(uploaded_files) > 20:
            st.error("最多只能上传20个音频文件")
        else:
            batch_config = {
                "batch_name": batch_name,
                "dimensions": selected_dims,
                "baseline_file": baseline_file
            }
            with st.spinner("正在提交批量任务..."):
                result = call_batch_submit_api(uploaded_files, batch_config)
                if result:
                    st.session_state["batch_id"] = result["batch_id"]
                    st.session_state["batch_status"] = None
                    st.session_state["batch_report"] = None
                    st.success(f"✅ 批量任务提交成功! Batch ID: {result['batch_id']}")

    if "batch_id" in st.session_state:
        st.markdown("---")
        batch_id = st.session_state["batch_id"]
        st.info(f"当前批次ID: `{batch_id}`")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 刷新状态", use_container_width=True):
                pass
        with col2:
            if st.button("❌ 清除当前批次", use_container_width=True):
                for key in ["batch_id", "batch_status", "batch_report"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        status = call_batch_status_api(batch_id)
        if status:
            progress = status["progress"]
            st.subheader("📊 处理进度")
            progress_bar = st.progress(progress["percentage"] / 100)
            st.metric(
                label="整体进度",
                value=f"{progress['completed']}/{progress['total']}",
                delta=f"{progress['percentage']:.1f}%"
            )

            st.subheader("📁 文件状态")
            status_df = pd.DataFrame([
                {
                    "文件名": f["filename"],
                    "状态": {
                        "pending": "⏳ 等待中",
                        "processing": "⚙️ 处理中",
                        "completed": "✅ 完成",
                        "failed": "❌ 失败"
                    }.get(f["status"], f["status"]),
                    "错误": f.get("error", "")
                }
                for f in status["files"]
            ])
            st.dataframe(status_df, use_container_width=True)

            if progress["percentage"] < 100:
                st.info("⏳ 正在处理中，页面将自动刷新...")
                time.sleep(3)
                st.rerun()
            else:
                st.success("✅ 所有文件处理完成!")

        st.subheader("📋 对比报告")
        report_result = call_batch_report_api(batch_id)
        if report_result["status"] == "processing":
            st.info("⏳ 报告生成中，请稍候...")
        elif report_result["status"] == "completed":
            report = report_result["data"]
            if "error" in report:
                st.error(f"报告生成失败: {report['error']}")
            else:
                meta = report.get("meta", {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("批次名称", meta.get("batch_name", ""))
                with col2:
                    st.metric("总文件数", meta.get("total_files", 0))
                with col3:
                    st.metric("成功数", meta.get("success_count", 0))
                with col4:
                    st.metric("总耗时", f"{meta.get('total_duration_seconds', 0):.1f}s")

                col_csv, col_json = st.columns(2)
                with col_csv:
                    csv_result = call_batch_report_api(batch_id, format="csv")
                    if csv_result["status"] == "completed":
                        st.download_button(
                            label="📊 下载CSV报告",
                            data=csv_result["data"],
                            file_name=f"batch_report_{batch_id}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                with col_json:
                    st.download_button(
                        label="📄 下载JSON报告",
                        data=json.dumps(report, ensure_ascii=False, indent=2),
                        file_name=f"batch_report_{batch_id}.json",
                        mime="application/json",
                        use_container_width=True
                    )

                comparison = report.get("comparison", {})
                if "emotion_distribution" in comparison:
                    st.markdown("---")
                    st.subheader("🎭 情感分布对比")
                    plot_batch_emotion_radar(report)
                    plot_batch_emotion_bar(report)

                    most_div = comparison["emotion_distribution"].get("most_divergent_pair", {})
                    if most_div.get("file1"):
                        f1 = report["file_summaries"].get(most_div["file1"], {}).get("filename", most_div["file1"])
                        f2 = report["file_summaries"].get(most_div["file2"], {}).get("filename", most_div["file2"])
                        st.info(f"🔍 情感分布差异最大的文件对: **{f1}** vs **{f2}** (JS散度: {most_div['js_divergence']:.4f})")

                if "valence_trend" in comparison:
                    st.markdown("---")
                    st.subheader("📈 效价趋势对比")
                    plot_valence_trend_comparison(report)

                    anomalous = comparison["valence_trend"].get("anomalous_files", [])
                    if anomalous:
                        names = [report["file_summaries"].get(f, {}).get("filename", f) for f in anomalous]
                        st.warning(f"⚠️ 趋势异常文件: {', '.join(names)}")

                    for fid, data in comparison["valence_trend"]["trends"].items():
                        fname = report["file_summaries"].get(fid, {}).get("filename", fid)
                        trend_cn = {"rising": "上升", "falling": "下降", "stationary": "平稳"}.get(data["trend"], data["trend"])
                        st.caption(f"📊 {fname}: 趋势={trend_cn}, 斜率={data['slope']:.4f}, 平均效价={data['mean_valence']:.3f}")

                if "arousal_pattern" in comparison:
                    st.markdown("---")
                    st.subheader("⚡ 唤醒度模式对比")
                    plot_arousal_pattern_comparison(report)

                    files_with_esc = comparison["arousal_pattern"].get("files_with_escalation", [])
                    if files_with_esc:
                        names = [report["file_summaries"].get(f, {}).get("filename", f) for f in files_with_esc]
                        st.warning(f"⚠️ 存在情绪激化区间的文件: {', '.join(names)}")

                if "speaker_similarity" in comparison:
                    st.markdown("---")
                    st.subheader("👥 说话人同步性对比")
                    plot_similarity_matrix(report)

                    sync_data = comparison["speaker_similarity"].get("speaker_sync", {})
                    for fid, data in sync_data.items():
                        fname = report["file_summaries"].get(fid, {}).get("filename", fid)
                        if data["is_dual_channel"]:
                            if data["synchronization"]:
                                sync = data["synchronization"]
                                level_cn = {"high": "高", "medium": "中", "low": "低"}.get(sync["alignment_level"], sync["alignment_level"])
                                st.caption(f"🎙️ {fname}: 同步分数={sync['synchronization_score']:.3f}, 等级={level_cn}")
                            else:
                                st.caption(f"🎙️ {fname}: 双通道但说话人数据不足")
                        else:
                            st.caption(f"🎙️ {fname}: 单通道，跳过同步性分析")

                if meta.get("baseline_file"):
                    st.markdown("---")
                    st.subheader("📊 基线偏差分析")
                    plot_baseline_deviation_heatmap(report)

                st.markdown("---")
                with st.expander("📄 查看完整报告JSON", expanded=False):
                    st.json(report)


def main():
    st.set_page_config(page_title="语音情感识别与情绪追踪", layout="wide")

    health = call_health_api()
    if health:
        st.sidebar.success(f"API服务运行中 v{health.get('version', '?')}")
        for model, loaded in health.get("models_loaded", {}).items():
            status = "✅" if loaded else "❌"
            st.sidebar.text(f"{model}: {status}")
    else:
        st.sidebar.error("API服务未连接")

    tab_single, tab_batch = st.tabs(["🎙️ 单文件分析", "📊 批量对比"])

    with tab_single:
        st.title("🎙️ 语音情感识别与多轮对话情绪追踪")

        st.sidebar.header("参数设置")
        model_mode = st.sidebar.selectbox("模型模式", ["svm", "rf", "wav2vec2"], index=0, key="single_model_mode")
        is_dual = st.sidebar.checkbox("双通道录音", value=False, key="single_is_dual")
        output_fmt = st.sidebar.selectbox("输出格式", ["full", "compact"], index=0, key="single_output_fmt")

        uploaded_file = st.file_uploader("上传音频文件 (WAV/MP3, 最大50MB)",
                                         type=["wav", "mp3"], key="single_upload")

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            st.audio(uploaded_file, format=uploaded_file.type if uploaded_file.type else "audio/wav")

            if "audio_raw" not in st.session_state or st.session_state.get("audio_filename") != uploaded_file.name:
                parsed_audio, parsed_sr = _parse_audio_bytes(file_bytes, uploaded_file.name)
                st.session_state["audio_raw"] = parsed_audio
                st.session_state["audio_sr"] = parsed_sr
                st.session_state["audio_filename"] = uploaded_file.name

            if st.button("🔍 开始分析", type="primary", key="single_analyze_btn"):
                with st.spinner("正在分析音频..."):
                    result = call_analyze_api(
                        file_bytes, uploaded_file.name,
                        model_mode, is_dual, output_fmt
                    )

                if result:
                    st.session_state["analysis_result"] = result
                    st.session_state["analyzed_at"] = datetime.now().isoformat()

        if "analysis_result" in st.session_state:
            result = st.session_state["analysis_result"]

            if not result.get("completed", True):
                st.warning(f"⚠️ 分析未完全完成: {result.get('incomplete_reason', '超时')}")

            st.header("📊 音频信息")
            info = result.get("audio_info", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("时长", f'{info.get("duration", 0):.1f}秒')
            with col2:
                st.metric("采样率", f'{info.get("sample_rate", 0)}Hz')
            with col3:
                st.metric("通道数", info.get("channels", 0))

            st.header("🎵 波形与情感分段")
            plot_waveform_with_emotions(
                result,
                audio_data=st.session_state.get("audio_raw"),
                sr=st.session_state.get("audio_sr")
            )

            st.header("📈 情绪轨迹曲线")
            plot_emotion_trajectory(result)

            st.header("🥧 情感分布")
            plot_emotion_pie(result)

            if is_dual:
                st.header("👥 双人对话情绪分析")
                plot_dual_speaker_trajectory(result)

            st.header("📋 逐句情感结果")
            render_sentence_table(result)

            st.header("📝 对话情绪摘要")
            col_summary, col_export = st.columns([3, 1])
            with col_summary:
                render_summary(result)
            with col_export:
                st.subheader("导出标注")
                render_export_buttons(
                    result,
                    st.session_state.get("audio_filename", "audio.wav"),
                    st.session_state.get("analyzed_at")
                )

            with st.expander("🔧 模型性能参考"):
                metrics = result.get("metrics", {})
                st.json(metrics)

            sentences = result.get("sentences", [])
            if sentences:
                st.header("🎧 片段试听")
                sent_idx = st.selectbox("选择片段", range(len(sentences)),
                                        format_func=lambda i: f"句{i+1} ({sentences[i]['start_time']:.1f}s-{sentences[i]['end_time']:.1f}s) "
                                                               f"- {EMOTION_LABELS_CN.get(sentences[i]['emotion'], sentences[i]['emotion'])}",
                                        key="single_sent_select")
                sel = sentences[sent_idx]
                audio_raw = st.session_state.get("audio_raw")
                audio_sr = st.session_state.get("audio_sr")
                if audio_raw is not None and audio_sr is not None:
                    start_sample = int(sel["start_time"] * audio_sr)
                    end_sample = int(sel["end_time"] * audio_sr)
                    end_sample = min(end_sample, len(audio_raw))
                    if start_sample < end_sample:
                        segment = audio_raw[start_sample:end_sample]
                        wav_bytes = _audio_segment_to_wav_bytes(segment, audio_sr)
                        st.audio(wav_bytes, format="audio/wav")
                    else:
                        st.warning("片段时间范围无效")
                else:
                    st.info(f"播放: {sel['start_time']:.1f}s - {sel['end_time']:.1f}s (原始音频未缓存，无法截取)")

                with st.expander("🎬 对话回放", expanded=False):
                    render_dialogue_playback(
                        result,
                        audio_raw=st.session_state.get("audio_raw"),
                        audio_sr=st.session_state.get("audio_sr"),
                        is_dual=is_dual
                    )

    with tab_batch:
        render_batch_page()


if __name__ == "__main__":
    main()
