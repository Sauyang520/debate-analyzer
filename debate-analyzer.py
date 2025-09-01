import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Don't load TensorFlow
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"  # Don't load Torchvision
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable TensorFlow CPU warnings

import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from io import StringIO
import hashlib
import torch
from torch.nn.functional import softmax
import numpy as np

# ==== Streamlit Config ====
st.set_page_config(page_title="Debate Analyzer", layout="wide")
st.title("Debate Analyzer")

# === Uniform figure size for all charts ===
FIGSIZE_UNI = (5.6, 3.6)

# ==== NLTK setup ====
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

# ==== Session state ====
for key, default in {
    "show_results": False,
    "analysis_text": "",
    "restored_text": "",
    "text_hash": None,
    "fallacy_threshold": 0.8,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------- TABS --------------------
tab_analyze, tab_learn = st.tabs(["Analyze", "Learn"])

# =========================================================
# =============== TAB 1: ANALYZE (existing) ===============
# =========================================================
with tab_analyze:
    # ==== Input section ====
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Transcript")
        typed_text = st.text_area("Transcript:", placeholder="Paste or type here...")
    with col2:
        st.subheader("or Upload TXT File")
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    # Threshold selector
    st.subheader("Fallacy Detection Settings")
    st.session_state.fallacy_threshold = st.slider(
        "Fallacy Score Threshold",
        min_value=0.0, max_value=1.0, step=0.05,
        value=0.8, format="%.2f"
    )

    candidate_text = ""
    if uploaded_file:
        try:
            candidate_text = uploaded_file.read().decode("utf-8").strip()
        except UnicodeDecodeError:
            candidate_text = uploaded_file.read().decode("latin-1").strip()
    elif typed_text:
        candidate_text = typed_text.strip()

    if st.button("Analyze", type="primary", disabled=not bool(candidate_text)):
        st.session_state.analysis_text = candidate_text
        st.session_state.show_results = True
        st.rerun()

    # ==== Cached Model Loaders ====
    @st.cache_resource(show_spinner=False)
    def load_punct_model():
        from deepmultilingualpunctuation import PunctuationModel
        return PunctuationModel()

    @st.cache_resource(show_spinner=False)
    def load_arg_classifier():
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_path = "3_model/deberta-v3"  # Local model path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return tokenizer, model

    @st.cache_resource(show_spinner=False)
    def load_fallacy_pipeline(
        model_src: str = "q3fer/distilbert-base-fallacy-classification",
        prefer_local_dir: str = "models/fallacy"
    ):
        from transformers import pipeline
        from pathlib import Path

        if Path(prefer_local_dir).is_dir():
            return pipeline(
                "text-classification",
                model=str(prefer_local_dir),
                tokenizer=str(prefer_local_dir),
                use_safetensors=True,
                device=0 if torch.cuda.is_available() else -1
            )
        return pipeline(
            "text-classification",
            model=model_src,
            tokenizer=model_src,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=None
        )

    # ==== Constants ====
    VALID_LABELS = ['C', 'G', 'W/Q', 'OTH']
    IDX2LABEL = {i: lab for i, lab in enumerate(VALID_LABELS)}

    # ==== Processing ====
    if st.session_state.show_results and st.session_state.analysis_text:
        raw_text = st.session_state.analysis_text
        current_hash = hashlib.md5(raw_text.encode("utf-8")).hexdigest()

        if st.session_state.text_hash != current_hash or not st.session_state.restored_text:
            with st.spinner("Restoring punctuation…"):
                restored = load_punct_model().restore_punctuation(raw_text)
            st.session_state.restored_text = restored
            st.session_state.text_hash = current_hash

        final_text = st.session_state.restored_text
        threshold = st.session_state.fallacy_threshold

        st.markdown("---")
        st.header("Results")

        sentences = nltk.sent_tokenize(final_text, language="english")
        st.write(f"Your debate transcript contains {len(sentences)} sentences!")

        # Argument classification
        with st.spinner("Classifying argument types…"):
            tokenizer, model = load_arg_classifier()
            preds, confs = [], []
            batch_size = 16

            def to_device(batch):
                return {k: v.cuda() for k, v in batch.items()} if torch.cuda.is_available() else batch

            for i in range(0, len(sentences), batch_size):
                enc = tokenizer(
                    sentences[i:i+batch_size],
                    padding=True, truncation=True,
                    return_tensors="pt", max_length=256
                )
                enc = to_device(enc)
                with torch.no_grad():
                    probs = softmax(model(**enc).logits, dim=1)
                    preds.extend(torch.argmax(probs, dim=1).cpu().tolist())
                    confs.extend(probs.max(dim=1).values.cpu().tolist())

        arg_labels = [IDX2LABEL[i] for i in preds]
        arg_conf = [round(c, 4) for c in confs]

        # Fallacy detection
        with st.spinner("Detecting logical fallacies…"):
            fallacy_pipe = load_fallacy_pipeline()
            fallacy_results = fallacy_pipe(
                sentences, truncation=True, padding=True, max_length=256, batch_size=16
            )

        def _pick(res):
            if isinstance(res, list):
                res = res[0]
            return res["label"], float(res["score"])

        fallacy_labels, fallacy_scores = zip(*[_pick(r) for r in fallacy_results]) if sentences else ([], [])

        # Combine results
        df = pd.DataFrame({
            "No.": range(1, len(sentences) + 1),
            "sentence": sentences,
            "argument_type": arg_labels,
            "confidence": arg_conf,
            "fallacy": list(fallacy_labels),
            "fallacy_score": [round(s, 4) for s in fallacy_scores]
        })
        df["fallacy_final"] = np.where(df["fallacy_score"] >= threshold, df["fallacy"], "None")

        # ===================== TABLE =====================
        st.subheader("Labelled Sentences")
        display_df = df[["No.", "sentence", "argument_type", "fallacy_final"]].rename(
            columns={"fallacy_final": "fallacy"}
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # KPIs
        total = len(df)
        any_fallacy = (df["fallacy_final"] != "None").sum()
        st.markdown(
            f"**Summary:** {total} sentences • {any_fallacy} with validated fallacies "
            f"({(any_fallacy/total*100 if total else 0):.1f}%)."
        )

        # ===================== Charts =====================
        st.subheader("Charts")

        # ROW 1
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig_wc, ax_wc = plt.subplots(figsize=FIGSIZE_UNI)
            wc = WordCloud(width=900, height=360, background_color="white",
                           stopwords=STOPWORDS).generate(final_text)
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            ax_wc.set_title("Word Cloud", fontsize=12, pad=6)
            st.pyplot(fig_wc, use_container_width=True)

        with r1c2:
            sent_word_lengths = df["sentence"].apply(lambda s: len(s.split()))
            fig_len, ax_len = plt.subplots(figsize=FIGSIZE_UNI)
            ax_len.hist(sent_word_lengths, bins=20)
            ax_len.set_title("Sentence Length Distribution", fontsize=12, pad=6)
            ax_len.set_xlabel("Length (words)")
            ax_len.set_ylabel("Frequency")
            st.pyplot(fig_len, use_container_width=True)

        # ROW 2
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            arg_counts = df["argument_type"].value_counts().sort_index()
            labels = arg_counts.index.tolist()
            sizes  = arg_counts.values.tolist()
            pie_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

            def autopct_with_count(pct):
                total = sum(sizes)
                count = int(round(pct/100.0 * total))
                return f"{pct:.0f}%\n({count})"

            fig_pie, ax_pie = plt.subplots(figsize=FIGSIZE_UNI)
            ax_pie.pie(sizes, labels=labels, autopct=autopct_with_count,
                       startangle=90, colors=pie_colors)
            ax_pie.axis("equal")
            ax_pie.set_title("Argument Type Distribution", fontsize=12, pad=6)
            st.pyplot(fig_pie, use_container_width=True)

        with r2c2:
            ct = pd.crosstab(df["argument_type"], df["fallacy_final"])
            fig_hm, ax_hm = plt.subplots(figsize=FIGSIZE_UNI)
            im = ax_hm.imshow(ct.values, aspect="auto", cmap="Blues", vmin=0)
            ax_hm.set_xticks(range(ct.shape[1]))
            ax_hm.set_xticklabels(ct.columns, rotation=30, ha="right", fontsize=8)
            ax_hm.set_yticks(range(ct.shape[0]))
            ax_hm.set_yticklabels(ct.index, fontsize=8)
            ax_hm.set_xlabel("Fallacy")
            ax_hm.set_ylabel("Argument Type")
            for i in range(ct.shape[0]):
                for j in range(ct.shape[1]):
                    ax_hm.text(j, i, str(ct.values[i, j]), ha="center", va="center", fontsize=8)
            fig_hm.colorbar(im, ax=ax_hm, shrink=0.8).set_label("Count")
            ax_hm.set_title("Argument Type × Fallacy Heatmap", fontsize=12, pad=6)
            st.pyplot(fig_hm, use_container_width=True)

        # ROW 3
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            fall_counts = df.loc[df["fallacy_final"] != "None", "fallacy_final"].value_counts()
            fig_fd, ax_fd = plt.subplots(figsize=FIGSIZE_UNI)
            if len(fall_counts) == 0:
                ax_fd.text(0.5, 0.5, "No validated fallacies", ha="center", va="center")
                ax_fd.axis("off")
            else:
                ax_fd.bar(fall_counts.index, fall_counts.values)
                for i, v in enumerate(fall_counts.values):
                    ax_fd.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
                ax_fd.set_xlabel("Fallacy")
                ax_fd.set_ylabel("Count")
                ax_fd.tick_params(axis="x", rotation=30, labelsize=8)
            ax_fd.set_title(f"Fallacy Distribution (≥ {threshold})", fontsize=12, pad=6)
            st.pyplot(fig_fd, use_container_width=True)

        with r3c2:
            fig_fs, ax_fs = plt.subplots(figsize=FIGSIZE_UNI)
            ax_fs.hist(df["fallacy_score"], bins=20)
            ax_fs.axvline(threshold, linestyle="--")
            ax_fs.set_title("Fallacy Score Distribution", fontsize=12, pad=6)
            ax_fs.set_xlabel("Fallacy Score")
            ax_fs.set_ylabel("Frequency")
            st.pyplot(fig_fs, use_container_width=True)

        # ------------------ Gemini (kept as in your version) ------------------
        import google.generativeai as genai
        
        # Load Gemini API key
        GEMINI_API_KEY = "AIzaSyDEvdpMOA85iX3_eYyHNseWAsq-iSO8XWw"
        genai.configure(api_key=GEMINI_API_KEY)
        
        advice = ""
        def get_gemini_recommendations(df, model_name="gemini-2.0-flash"):
            # Build a compact summary of results to send
            summary = []
            for _, row in df.iterrows():
                summary.append(f"Sentence: {row['sentence']}\n"
                               f"Argument Type: {row['argument_type']}\n"
                               f"Fallacy: {row['fallacy_final']}\n")
            summary_text = "\n".join(summary[:50])
        
            prompt = f"""
            You are an experienced debate coach. Review the following transcript with identified argument types and fallacies.

            Tasks:
            1. Critically evaluate the overall flow and coherence of the debate.
            2. Identify patterns in fallacy use and explain their impact on persuasiveness.
            3. Provide targeted advice on how to strengthen each argument type (Claim, Grounds, Warrant/Qualifier).
            4. Conclude with actionable debate strategies in concise bullet points.

            Transcript analysis:
            {summary_text}
            """
        
            # model = genai.GenerativeModel(model_name)
            # response = model.generate_content(prompt)
            # return response.text if response else "No recommendations generated."
            return ""
        
        if GEMINI_API_KEY:
            st.subheader("Gemini Flash 2.0 Recommendations")
            with st.spinner("Generating recommendations..."):
                advice = get_gemini_recommendations(df)
            st.write(advice)

        # ------------------ Download ------------------
        st.subheader("Download")
        ordered_cols = ["No.", "sentence", "argument_type", "confidence",
                        "fallacy", "fallacy_score", "fallacy_final"]
        csv_buf = StringIO()
        df[ordered_cols].to_csv(csv_buf, index=False)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download Labelled Sentences",
                               data=csv_buf.getvalue(),
                               file_name="analyzed_transcript.csv",
                               mime="text/csv")

        with c2:
            st.download_button("Download Gemini Feedback",
                               data=advice,
                               file_name="gemini_feedback.txt",
                               mime="text/plain")

    else:
        st.info("Type or upload text, then click **Analyze**.")

# =========================================================
# =============== TAB 2: LEARN (new content) ==============
# =========================================================
with tab_learn:
    st.header("Toulmin Model & Fallacies")

    st.subheader("Toulmin Argument Model")
    st.image("toulmin_argument.png", caption="Toulmin Argument", width=700)
    st.markdown("""
**Components**
- **Claim (C):** The point you want the audience to accept.
- **Grounds (G):** Facts or reasons that support the claim.
- **Warrant (W):** The logic that links the grounds to the claim.
- **Qualifier (Q):** Limits or strength of the claim (e.g., *probably*, *often*).
- **Backing (B):** Extra support for the warrant (data, theory, authority).
- **Rebuttal (R):** When or why the claim might not hold; counter-cases.

**Example**
- **Claim:** Schools should start later.
- **Grounds:** Studies show teens learn better after 9 a.m.
- **Warrant:** Policy should follow evidence that improves learning.
- **Backing:** Sleep research on circadian rhythms.
- **Qualifier:** Especially for secondary schools.
- **Rebuttal:** Except where transport or safety cannot be solved yet.
""")

    st.subheader("Fallacies")
    st.markdown("""
**1) Circular reasoning** - The claim is used as its own proof.  
*“It's true because it's true.”*

**2) Fallacy of logic** - The reasoning does not follow or is inconsistent (a general logic error).  
*The conclusion doesn't follow from the reasons.*

**3) Equivocation** - A key word changes meaning mid-argument.  
*“fair” meaning “just” vs “pleasant”.*

**4) Fallacy of credibility** - Leaning on status/authority instead of evidence, or using a biased/irrelevant expert.  
*“A famous YouTuber says so.”*

**5) Ad populum** - “Everyone believes it” → so it must be true.  
*Popularity ≠ proof.*

**6) Fallacy of extension** - Extending or exaggerating the opponent's point to attack it (straw-man style).  
*“You want rules? So you want a police state.”*

**7) Intentional** - Arguing about intent to prove truth or excuse harm; assuming good/bad intent decides the claim.  
*“They meant well, so the policy works.”*

**8) Faulty generalization** - Drawing a big rule from too little or unrepresentative data.  
*One story ⇒ a universal rule.*

**9) Appeal to emotion** - Using fear, pity, anger, pride as the main “reason.”  
*Feelings replace evidence.*

**10) Fallacy of relevance** - Bringing in points that are off-topic or distract (red herring/whataboutism).  
*Changes subject instead of answering.*

**11) False dilemma** - Pretending there are only two choices when more exist.  
*“With us or against us.”*

**12) Ad hominem** - Attacking the person instead of the argument.  
*“You're lazy, so your idea is wrong.”*

**13) False causality** - Assuming cause from timing or correlation.  
*After X came Y ⇒ X caused Y.*

**14) Miscellaneous** - Mixed or minor errors that don't fit the above labels; still weaken clarity or logic.
""")

    st.subheader("How to Strengthen Your Arguments")
    st.markdown("""
- State a clear **claim**, give solid **grounds**, and make the **warrant** explicit.
- Use **qualifiers** (avoid absolute words like *always/never*).
- Add **backing** from credible, relevant sources.
- Address **rebuttals** fairly and show why your claim still stands.
- Replace emotional lines with **measurable impacts** and **clear comparisons**.
""")

