import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
import json

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


def plot_waveform_with_emotions(result):
    sentences = result.get("sentences", [])
    if not sentences:
        st.info("没有检测到语音段")
        return

    fig = go.Figure()
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
        title="语音分段与情感标注",
        xaxis_title="时间 (秒)",
        yaxis_title="",
        showlegend=True,
        height=300,
        yaxis=dict(showticklabels=False)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_emotion_trajectory(result):
    sentences = result.get("sentences", [])
    summary = result.get("dialogue_summary", {})
    if not sentences:
        return

    times = [(s["start_time"] + s["end_time"]) / 2 for s in sentences]
    valences = [s["valence"] for s in sentences]
    arousals = [s["arousal"] for s in sentences]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("效价 (Valence) 轨迹", "唤醒度 (Arousal) 轨迹"))

    fig.add_trace(go.Scatter(
        x=times, y=valences, mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=8),
        name='效价'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=times, y=arousals, mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=8),
        name='唤醒度'
    ), row=2, col=1)

    for tp in summary.get("turning_points", []):
        fig.add_vline(x=tp["time"], line_dash="dash", line_color="orange",
                      annotation_text="转折", row=1, col=1)

    for esc in summary.get("escalation_intervals", []):
        fig.add_vrect(x0=esc["start_time"], x1=esc["end_time"],
                      fillcolor="red", opacity=0.1,
                      annotation_text="激化", row=2, col=1)

    fig.update_layout(height=500, title_text="情绪轨迹曲线")
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


def main():
    st.set_page_config(page_title="语音情感识别与情绪追踪", layout="wide")
    st.title("🎙️ 语音情感识别与多轮对话情绪追踪")

    health = call_health_api()
    if health:
        st.sidebar.success(f"API服务运行中 v{health.get('version', '?')}")
        for model, loaded in health.get("models_loaded", {}).items():
            status = "✅" if loaded else "❌"
            st.sidebar.text(f"{model}: {status}")
    else:
        st.sidebar.error("API服务未连接")

    st.sidebar.header("参数设置")
    model_mode = st.sidebar.selectbox("模型模式", ["svm", "rf", "wav2vec2"], index=0)
    is_dual = st.sidebar.checkbox("双通道录音", value=False)
    output_fmt = st.sidebar.selectbox("输出格式", ["full", "compact"], index=0)

    uploaded_file = st.file_uploader("上传音频文件 (WAV/MP3, 最大50MB)",
                                     type=["wav", "mp3"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        st.audio(uploaded_file, format=uploaded_file.type if uploaded_file.type else "audio/wav")

        if st.button("🔍 开始分析", type="primary"):
            with st.spinner("正在分析音频..."):
                result = call_analyze_api(
                    file_bytes, uploaded_file.name,
                    model_mode, is_dual, output_fmt
                )

            if result:
                st.session_state["analysis_result"] = result

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
        plot_waveform_with_emotions(result)

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
        render_summary(result)

        with st.expander("🔧 模型性能参考"):
            metrics = result.get("metrics", {})
            st.json(metrics)

        sentences = result.get("sentences", [])
        if sentences:
            st.header("🎧 片段试听")
            sent_idx = st.selectbox("选择片段", range(len(sentences)),
                                    format_func=lambda i: f"句{i+1} ({sentences[i]['start_time']:.1f}s-{sentences[i]['end_time']:.1f}s) "
                                                           f"- {EMOTION_LABELS_CN.get(sentences[i]['emotion'], sentences[i]['emotion'])}")
            st.info(f"播放: {sentences[sent_idx]['start_time']:.1f}s - {sentences[sent_idx]['end_time']:.1f}s")


if __name__ == "__main__":
    main()
