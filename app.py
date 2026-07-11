from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from bandscribe_core import (
    build_analysis,
    demucs_is_available,
    ensure_dirs,
    record_harmony_feedback,
    save_upload,
)
from bandscribe_ml import KEY_LABELS, PROGRESSION_LABELS
from train_models import train


ROOT = Path(__file__).resolve().parent


STAGES = [
    ("阶段 0", "上传音频、保存到 outputs/uploads、展示原始试听。"),
    ("阶段 1", "尝试 demucs htdemucs_6s 分离；失败时用原音频占位。"),
    ("阶段 2", "生成架子鼓谱、节奏吉他和弦、主音吉他六线谱和键盘五线谱。"),
    ("阶段 3", "为鼓/和弦/旋律/全曲草稿生成 MIDI 与 WAV 试听。"),
    ("阶段 4", "生成至少 3 条改编思路，并给出验证方法。"),
    ("阶段 5", "训练调性/和弦进行模型，显示置信度并收集人工纠错。"),
]


def main() -> None:
    st.set_page_config(page_title="BandScribe", page_icon="BS", layout="wide")
    ensure_dirs(ROOT)

    st.title("BandScribe")
    st.caption("上传一段乐队音频，先跑通分离、粗谱、改编和试听；准确率先按 40% 目标推进。")

    with st.sidebar:
        st.header("Checkpoint 验证")
        for title, note in STAGES:
            st.markdown(f"**{title}**  \n{note}")
        st.divider()
        st.write("热重载：`.streamlit/config.toml` 已开启保存后自动 rerun。")
        st.write(f"demucs 状态：{'可用' if demucs_is_available() else '未安装，使用降级模式'}")
        model_path = ROOT / "models" / "harmony_model.npz"
        st.write(f"和声模型：{'已训练' if model_path.exists() else '未训练'}")
        if st.button("重新训练和声模型", width="stretch"):
            with st.spinner("正在生成合成样本并训练..."):
                st.session_state["training_metrics"] = train(model_path)
            st.success("训练完成，下一次分析会自动加载新模型。")
        training_metrics = st.session_state.get("training_metrics")
        if training_metrics:
            st.caption(
                f"合成留出集：调性 {training_metrics['test']['key_accuracy']:.0%}，"
                f"进行 {training_metrics['test']['progression_accuracy']:.0%}；不代表真实歌曲准确率。"
            )
        use_demucs = st.toggle("尝试真实 demucs 分离", value=True)
        if st.button("清空当前页面结果"):
            st.session_state.pop("result", None)
            st.rerun()

    uploaded = st.file_uploader("上传音频", type=["wav", "mp3", "flac", "m4a", "ogg", "aac"])
    if uploaded is None:
        st.info("先上传一首歌或排练录音。WAV 能得到更稳定的节奏估计；其他格式也会先走启发式。")
        return

    data = uploaded.getvalue()
    st.subheader("原始音频")
    st.audio(data, format=uploaded.type or "audio/wav")
    st.write({"文件名": uploaded.name, "大小": f"{len(data) / 1_000_000:.2f} MB"})

    if st.button("开始分析", type="primary"):
        with st.status("阶段 0：保存上传文件", expanded=True) as status:
            job_id, upload_path = save_upload(ROOT, uploaded.name, data)
            st.write(f"已保存：`{upload_path}`")

            status.update(label="阶段 1：音轨分离或降级占位")
            result = build_analysis(ROOT, job_id, Path(upload_path), use_demucs=use_demucs)
            st.session_state["result"] = result
            st.write(f"分离模式：`{result['separation']['mode']}`")

            status.update(label="阶段 2/3/4：谱面、MIDI/WAV、改编思路已生成", state="complete")

    result = st.session_state.get("result")
    if result:
        render_result(result)


def render_result(result: dict) -> None:
    if "notation" not in result:
        st.warning("这是热重载前的旧分析结果，请再次点击“开始分析”生成乐器谱。")
        return
    metrics = result["metrics"]
    st.divider()
    st.subheader("一眼结论")
    cols = st.columns(5)
    cols[0].metric("估计 BPM", metrics["bpm"])
    cols[1].metric("估计调性", metrics["key"])
    cols[2].metric("能量", f"{metrics['energy']:.2f}")
    cols[3].metric("密度", f"{metrics['density']:.2f}")
    cols[4].metric("来源", metrics["source"])
    for warning in metrics["warnings"]:
        st.warning(warning)

    tabs = st.tabs(["音轨分离", "鼓谱", "和弦", "旋律", "改编思路", "验证方法"])

    with tabs[0]:
        st.write(result["separation"]["log"])
        stem_labels = {
            "drums": "鼓组轨",
            "rhythm_guitar": "节奏吉他候选轨",
            "lead_or_keys": "主音/键盘候选轨",
            "bass": "贝斯轨",
            "vocals": "人声轨",
            "full_mix": "原始全曲",
        }
        for key, path in result["separation"]["stems"].items():
            st.markdown(f"**{stem_labels.get(key, key)}**")
            st.audio(path)

    with tabs[1]:
        drum_views = st.tabs(["架子鼓谱", "事件表"])
        with drum_views[0]:
            st.code(result["notation"]["drum_score"], language=None)
            download_score(result["notation"]["drum_score_path"], "下载架子鼓谱", f"drum_{result['job_id']}")
        with drum_views[1]:
            st.dataframe(pd.DataFrame(result["drums"]["rows"]), width="stretch", hide_index=True)
        audio_and_midi(result["artifacts"]["drum_wav"], result["artifacts"]["drum_midi"], "鼓组节奏")

    with tabs[2]:
        st.markdown("**节奏吉他/键盘和弦进行**")
        st.dataframe(pd.DataFrame(result["chords"]["rows"]), width="stretch", hide_index=True)
        harmony = result["harmony_model"]
        st.caption(
            f"来源：{harmony['source']} | 调性置信度：{harmony['key_confidence']:.0%} | "
            f"进行置信度：{harmony['progression_confidence']:.0%}"
        )
        audio_and_midi(result["artifacts"]["chord_wav"], result["artifacts"]["chord_midi"], "和弦进行")

        st.markdown("**成员校正**")
        feedback_cols = st.columns(2)
        predicted_key = result["metrics"]["key"]
        key_index = KEY_LABELS.index(predicted_key) if predicted_key in KEY_LABELS else 0
        predicted_progression = harmony.get("progression")
        progression_index = (
            PROGRESSION_LABELS.index(predicted_progression)
            if predicted_progression in PROGRESSION_LABELS
            else 0
        )
        corrected_key = feedback_cols[0].selectbox(
            "正确调性",
            KEY_LABELS,
            index=key_index,
            key=f"feedback_key_{result['job_id']}",
        )
        corrected_progression = feedback_cols[1].selectbox(
            "正确和弦进行",
            PROGRESSION_LABELS,
            index=progression_index,
            key=f"feedback_progression_{result['job_id']}",
        )
        if st.button("保存这次校正", key=f"save_feedback_{result['job_id']}"):
            feedback_path = record_harmony_feedback(
                ROOT,
                result,
                corrected_key,
                corrected_progression,
            )
            st.success(f"已写入训练反馈：{feedback_path}")

    with tabs[3]:
        melody_views = st.tabs(["主音吉他六线谱", "键盘五线谱", "音符事件"])
        with melody_views[0]:
            st.code(result["notation"]["guitar_tab"], language=None)
            download_score(result["notation"]["guitar_tab_path"], "下载主音吉他六线谱", f"tab_{result['job_id']}")
        with melody_views[1]:
            st.code(result["notation"]["keyboard_score"], language=None)
            download_score(result["notation"]["keyboard_score_path"], "下载键盘五线谱", f"keys_{result['job_id']}")
        with melody_views[2]:
            st.dataframe(pd.DataFrame(result["melody"]["rows"]), width="stretch", hide_index=True)
        audio_and_midi(result["artifacts"]["melody_wav"], result["artifacts"]["melody_midi"], "旋律")

    with tabs[4]:
        st.dataframe(pd.DataFrame(result["ideas"]), width="stretch", hide_index=True)
        st.markdown("**全曲草稿试听**")
        audio_and_midi(result["artifacts"]["full_wav"], result["artifacts"]["full_midi"], "全曲草稿")

    with tabs[5]:
        st.markdown(
            """
1. 阶段 0：上传音频后，应看到原始播放器和 `outputs/uploads/<job_id>/original.*`。
2. 阶段 1：如果已安装 demucs，应看到 drums/guitar/piano/bass/vocals 等 WAV；否则页面显示降级模式，所有轨道仍可试听原始音频。
3. 阶段 2：鼓谱、和弦表、旋律表必须都有内容，且乐队成员能直接按表试奏。
4. 阶段 3：每个谱面下方都必须有 WAV 试听和 MIDI 下载；全曲草稿也必须可试听。
5. 阶段 4：改编思路至少 3 条，并能对应到鼓、和弦、旋律或全曲草稿试听。
6. 阶段 5：侧栏显示和声模型已训练；和弦页显示模型来源、置信度，并可保存成员校正。
7. 开发热重载：运行 `streamlit run app.py` 后，保存任意源码文件应自动 rerun。
            """
        )


def audio_and_midi(wav_path: str, midi_path: str, label: str) -> None:
    st.audio(wav_path)
    path = Path(midi_path)
    st.download_button(
        f"下载 {label} MIDI",
        data=path.read_bytes(),
        file_name=path.name,
        mime="audio/midi",
        width="stretch",
    )


def download_score(path: str, label: str, key: str) -> None:
    score_path = Path(path)
    st.download_button(
        label,
        data=score_path.read_bytes(),
        file_name=score_path.name,
        mime="text/plain; charset=utf-8",
        key=key,
        width="stretch",
    )

if __name__ == "__main__":
    main()







