# app.py
import numpy as np
import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.graph_objects as go


# ==============================
# 0. 页面基本配置
# ==============================
st.set_page_config(
    page_title="Emotional Arc 情绪轨迹可视化",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Emotional Arc 情绪轨迹可视化")
st.write(
    "你可以上传一个 `.txt` 文本文件，或者直接把文章 / 小说片段粘贴进来，"
    "我们会帮你分析从头到尾的情绪变化，并画出一条“情绪轨迹”。"
)

st.info(
    "简单理解：我们把整篇文本切成很多小段，一段一段打分（0 ≈ 负向，1 ≈ 正向），"
    "然后按照阅读顺序连成一条线。鼠标移动到线上任何一个点，都可以看到该位置的情绪分数和片段摘要。"
)

# ==============================
# 1. 设备 & 模型缓存（只加载一次）
# ==============================
@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, device


tokenizer, model, device = load_model_and_tokenizer()
st.sidebar.success(f"情感模型已就绪，当前设备：{device}")


# ==============================
# 2. 会话状态：保存分析结果
# ==============================
if "arc_data" not in st.session_state:
    st.session_state.arc_data = None  # 保存最近一次的分析结果


# ==============================
# 3. 工具函数：滑动窗口 / 情感得分 / 重采样
# ==============================
def sliding_windows(text: str, window_size: int = 80, step: int = 60):
    """
    把整段文本按“字符”切成一小段一小段。
    window_size: 每个窗口包含的字符数
    step: 每次前进的步长（字符）
    """
    windows = []
    positions = []

    n = len(text)
    if n == 0:
        return windows, positions
    if n <= window_size:
        windows.append(text)
        positions.append(0)
        return windows, positions

    for i in range(0, n, step):
        window = text[i: i + window_size]
        if not window:
            break
        windows.append(window)
        positions.append(i)
        if len(window) < window_size:
            break

    return windows, positions


def sentiment_scores(sent_list, batch_size: int = 32, max_length: int = 64):
    """
    批量算情感得分（0~1 的“偏正面概率”）。
    """
    all_scores = []
    if not sent_list:
        return all_scores

    for i in range(0, len(sent_list), batch_size):
        batch = sent_list[i: i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pos_probs = probs[:, 1].detach().cpu().numpy().tolist()

        all_scores.extend(pos_probs)

    return all_scores


def resample_series(values, target_len: int = 20):
    """
    把原始情感序列“压缩 / 拉伸”到固定长度 target_len，
    方便不同长度文本之间做大致对比。
    """
    if target_len <= 0:
        raise ValueError("target_len must be positive")

    if len(values) == 0:
        return [0.0] * target_len, np.linspace(0, 1, target_len).tolist()
    if len(values) == 1:
        return [float(values[0])] * target_len, np.linspace(0, 1, target_len).tolist()

    values = np.array(values, dtype=float)
    n = len(values)

    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    new_values = np.interp(x_new, x_old, values)

    return new_values.tolist(), x_new.tolist()


# ==============================
# 4. 侧边栏参数设置（尽量人话）
# ==============================
st.sidebar.header("🔧 分析参数（如不确定，保持默认即可）")

window_size = st.sidebar.number_input(
    "每个片段的长度（字符数）",
    min_value=10,
    max_value=2000,
    value=80,
    step=10,
    help="可以理解为“一个镜头”的长度。数字越大，每段内容越长，情绪曲线越“粗颗粒”。",
)

step_size = st.sidebar.number_input(
    "片段之间的间隔（步长）",
    min_value=1,
    max_value=2000,
    value=60,
    step=5,
    help="每次往前推进多少个字符去取下一段。步长越小，曲线越平滑，但计算稍慢。",
)

arc_len = st.sidebar.number_input(
    "重采样点数（标准化情绪轨迹的长度）",
    min_value=5,
    max_value=200,
    value=20,
    step=1,
    help="比如设置为 20，就会把整篇文本的情绪走势“压缩”为 20 个关键节点。",
)

st.sidebar.markdown("---")
advanced = st.sidebar.checkbox("展开高级设置（一般不用动）", value=False)

if advanced:
    batch_size = st.sidebar.number_input(
        "批量大小 batch_size",
        min_value=1,
        max_value=128,
        value=32,
        step=1,
        help="一次送入模型计算的片段数量。越大越快，但显存 / 内存占用也会增加。",
    )
    max_length = st.sidebar.number_input(
        "每段转换成 token 后的最长长度 max_length",
        min_value=16,
        max_value=256,
        value=64,
        step=8,
        help="防止极长片段导致计算太慢或溢出。一般保持默认即可。",
    )
else:
    batch_size = 32
    max_length = 64
    st.sidebar.caption("高级参数已使用推荐默认值，如出现性能问题再来调整即可。")


# ==============================
# 5. 文本输入区域：上传文件 or 文本框
# ==============================
st.subheader("1️⃣ 准备文本")

col_file, col_text = st.columns(2)

with col_file:
    uploaded_file = st.file_uploader(
        "方式一：上传 `.txt` 文件",
        type=["txt"],
        help="支持 UTF-8 或 GBK 编码的纯文本文件。",
    )

with col_text:
    text_input = st.text_area(
        "方式二：直接粘贴文本内容",
        value="",
        height=220,
        placeholder="例如：一段小说、一篇文章、长评论、长微博等……",
    )

# 读取文件内容
file_text = ""
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    try:
        file_text = bytes_data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            file_text = bytes_data.decode("gbk")
        except UnicodeDecodeError:
            st.error("暂时无法识别这个 txt 文件的编码，请确认为 UTF-8 或 GBK。")

# 优先使用上传文件，其次是文本框
final_text = file_text.strip() if file_text else text_input.strip()

MAX_CHARS = 20000
if not final_text:
    st.info("请先上传一个 txt 文件，或者在右侧文本框中输入 / 粘贴一段文本。")
else:
    if file_text:
        st.success(
            f"已使用上传文件：**{uploaded_file.name}**，"
            f"文本长度约 **{len(final_text)}** 个字符。"
        )
    else:
        st.success(f"已使用文本框中的内容，文本长度约 **{len(final_text)}** 个字符。")

    if len(final_text) > MAX_CHARS:
        st.warning(
            f"当前文本长度为 {len(final_text)} 个字符，已经比较长了。"
            "分析可能会稍慢，如果只是想试试效果，可以先截取其中一段来玩。"
        )


# ==============================
# 6. 情感分析主函数（带缓存）
# ==============================
@st.cache_data(show_spinner=False)
def compute_emotional_arc(
    text: str,
    window_size: int,
    step_size: int,
    arc_len: int,
    batch_size: int,
    max_length: int,
):
    windows, positions = sliding_windows(text, window_size, step_size)
    scores = sentiment_scores(windows, batch_size=batch_size, max_length=max_length)
    arc_scores, arc_x = resample_series(scores, target_len=arc_len)
    return positions, scores, arc_x, arc_scores, windows


# ==============================
# 7. 点击按钮开始分析（更新 session_state）
# ==============================
st.subheader("2️⃣ 开始分析")

run_btn = st.button("🚀 生成情绪轨迹", disabled=(not final_text))

if run_btn and final_text:
    with st.spinner("模型正在认真阅读你的文本并打分，请稍候…"):
        positions, scores, arc_x, arc_scores, windows = compute_emotional_arc(
            final_text,
            window_size=window_size,
            step_size=step_size,
            arc_len=arc_len,
            batch_size=batch_size,
            max_length=max_length,
        )

    if not positions:
        st.warning("没有得到任何有效片段，可能是窗口设置太大或者文本太短，可以调整参数再试试。")
        st.session_state.arc_data = None
    else:
        st.session_state.arc_data = {
            "final_text_len": len(final_text),
            "positions": positions,
            "scores": scores,
            "arc_x": arc_x,
            "arc_scores": arc_scores,
            "windows": windows,
        }
        st.success("情绪分析完成 ✅ 下滑查看情绪轨迹可视化。")


# ==============================
# 8. 展示结果：原始弧线 + 重采样弧线
# ==============================
arc_data = st.session_state.arc_data

if arc_data is not None:
    positions = arc_data["positions"]
    scores = arc_data["scores"]
    arc_x = arc_data["arc_x"]
    arc_scores = arc_data["arc_scores"]
    windows = arc_data["windows"]
    total_len = arc_data["final_text_len"]

    if not positions:
        st.warning("当前没有有效的分析结果，请检查文本或参数后重新分析。")
    else:
        scores_arr = np.array(scores)
        pos_arr = np.array(positions)

        avg_score = float(scores_arr.mean())
        min_score = float(scores_arr.min())
        max_score = float(scores_arr.max())
        min_idx = int(scores_arr.argmin())
        max_idx = int(scores_arr.argmax())
        min_pos = positions[min_idx]
        max_pos = positions[max_idx]

        # ---- 8.1 整体情绪概览 ----
        st.subheader("3️⃣ 整体情绪小结")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("整体平均情绪", f"{avg_score:.3f}")
        with col_b:
            st.metric("全篇最低情绪", f"{min_score:.3f}", help=f"大约出现在字符位置 {min_pos}")
        with col_c:
            st.metric("全篇最高情绪", f"{max_score:.3f}", help=f"大约出现在字符位置 {max_pos}")

        # ---- 8.2 情绪轨迹可视化（原始 + 重采样，用 tabs 切换）----
        st.subheader("4️⃣ 情绪轨迹可视化")

        tab_raw, tab_resampled = st.tabs(["原始情绪轨迹", "重采样后的标准化轨迹"])

        # 为 tooltip 准备简短摘要（避免太长）
        snippets = []
        for w in windows:
            s = w[:50]
            if len(w) > 50:
                s += "..."
            snippets.append(s)

        # --- Tab 1: 原始情绪轨迹 ---
        with tab_raw:
            st.markdown("**按文本实际位置绘制的情绪轨迹**（横轴是字符起始位置，纵轴是情绪分数）。")

            fig_raw = go.Figure()

            # 主线
            fig_raw.add_trace(
                go.Scatter(
                    x=positions,
                    y=scores,
                    mode="lines+markers",
                    name="情绪轨迹",
                    line=dict(color="#4F81BD", width=2),
                    marker=dict(color="#4F81BD", size=6),
                    customdata=[[i, snippets[i]] for i in range(len(positions))],
                    hovertemplate=(
                        "<b>片段 #%{customdata[0]}</b><br>"
                        "起始位置：%{x}<br>"
                        "情绪分数：%{y:.3f}<br>"
                        "片段摘要：%{customdata[1]}"
                    )))