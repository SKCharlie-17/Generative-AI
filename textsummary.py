from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """The global shift toward renewable energy is accelerating,
driven by concerns over climate change and energy security. Solar
and wind power installations have reached record levels worldwide."""

summary = summarizer(text, max_length=30, min_length=10, do_sample=False)
print(summary[0])


# What is Text Summarization?

# Text summarization is the process of generating a shorter version of a text that preserves its main meaning, key ideas, and intent.

# It’s a form of text generation, where the model takes a long input (like an article, report, or paragraph) and generates a concise summary, just like a human would do.


# Two Main Types of Text Summarization
# 1. Extractive Summarization

# The model selects important sentences or phrases directly from the original text.

# It does not generate new sentences; it simply extracts and rearranges existing content.

# Example:
# Input:

# "The Indian Space Research Organisation (ISRO) launched the Chandrayaan-3 mission to explore the Moon's south pole. The mission aims to demonstrate safe landing and roving on the lunar surface."
# Extractive summary:
# "ISRO launched Chandrayaan-3 to explore the Moon's south pole."

# . Abstractive Summarization

# The model understands the text and generates new sentences that express the same meaning — like how humans paraphrase.

# It’s a truly generative approach.

# Example:
# Input:

# "The Indian Space Research Organisation (ISRO) launched the Chandrayaan-3 mission to explore the Moon's south pole. The mission aims to demonstrate safe landing and roving on the lunar surface."
# Abstractive summary:
# "ISRO’s Chandrayaan-3 mission targets a soft landing on the Moon’s south pole."

# 🧩 Common models:

# Transformer-based models like T5, BART, PEGASUS, GPT, LLaMA, etc.

# ⚙️ How It Works in Generative AI

# Generative AI models (like GPT, BART, T5) are trained on large datasets of (document, summary) pairs.

# Input: A long text.

# Encoder: Understands the meaning of the input text.

# Decoder: Generates a shorter version that preserves meaning.

# Training objective: Minimize the difference between generated summary and reference summary.

# Applications of Text Summarization

# News summarization – Summarizing news articles.

# Research summarization – Summarizing scientific papers or abstracts.

# Legal/medical summarization – Condensing long legal documents or patient histories.

# Customer service – Summarizing support tickets or chat transcripts.

# Meeting summarization – Summarizing transcripts from meetings or lectures.


# This line imports the pipeline function from the Hugging Face Transformers library.

# The Transformers library provides pre-trained models for:

# Text summarization

# Text classification

# Sentiment analysis

# Translation

# Image classification

# Question answering

# Text generation (GPT-like models)

# Many more…

# The pipeline function is a shortcut tool that makes it very easy to use pre-trained models without needing to write complex code.

# ✅ Why use pipeline?

# Because it:

# ✔ Loads a pre-trained model automatically

# You don’t need to manually load the tokenizer and model.

# ✔ Automatically handles tokenization

# It converts your text to tokens internally.

# ✔ Runs the model



# ✔ Returns clean, human-readable output


# Text summarization
# Text summarization_
# What is Text Summarization?

# Text summarization is the process of generating a shorter version of a text that preserves its main meaning, key ideas, and intent.

# It’s a form of text generation, where the model takes a long input (like an article, report, or paragraph) and generates a concise summary, just like a human would do.

# Two Main Types of Text Summarization

# Extractive Summarization
# The model selects important sentences or phrases directly from the original text.

# It does not generate new sentences; it simply extracts and rearranges existing content.

# Example: Input:

# "The Indian Space Research Organisation (ISRO) launched the Chandrayaan-3 mission to explore the Moon's south pole. The mission aims to demonstrate safe landing and roving on the lunar surface." Extractive summary: "ISRO launched Chandrayaan-3 to explore the Moon's south pole."

# . Abstractive Summarization

# The model understands the text and generates new sentences that express the same meaning — like how humans paraphrase.

# It’s a truly generative approach.

# Example: Input:

# "The Indian Space Research Organisation (ISRO) launched the Chandrayaan-3 mission to explore the Moon's south pole. The mission aims to demonstrate safe landing and roving on the lunar surface." Abstractive summary: "ISRO’s Chandrayaan-3 mission targets a soft landing on the Moon’s south pole."

# 🧩 Common models:

# Transformer-based models like T5, BART, PEGASUS, GPT, LLaMA, etc.

# ⚙️ How It Works in Generative AI

# Generative AI models (like GPT, BART, T5) are trained on large datasets of (document, summary) pairs.

# Input: A long text.

# Encoder: Understands the meaning of the input text.

# Decoder: Generates a shorter version that preserves meaning.

# Training objective: Minimize the difference between generated summary and reference summary.

# Applications of Text Summarization

# News summarization – Summarizing news articles.

# Research summarization – Summarizing scientific papers or abstracts.

# Legal/medical summarization – Condensing long legal documents or patient histories.

# Customer service – Summarizing support tickets or chat transcripts.

# Meeting summarization – Summarizing transcripts from meetings or lectures.


# [ ]
# from transformers import pipeline

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# text = """The global shift toward renewable energy is accelerating,
# driven by concerns over climate change and energy security. Solar
# and wind power installations have reached record levels worldwide."""

# summary = summarizer(text, max_length=30, min_length=10, do_sample=False)
# print(summary[0])

# Device set to use cpu
# {'summary_text': 'Solar and wind power installations have reached record levels worldwide. The global shift toward renewable energy is accelerating.'}

# [ ]
# #from transformers import pipeline
# This line imports the pipeline function from the Hugging Face Transformers library.

# The Transformers library provides pre-trained models for:

# Text summarization

# Text classification

# Sentiment analysis

# Translation

# Image classification

# Question answering

# Text generation (GPT-like models)

# Many more…

# The pipeline function is a shortcut tool that makes it very easy to use pre-trained models without needing to write complex code.

# ✅ Why use pipeline?

# Because it:

# ✔ Loads a pre-trained model automatically

# You don’t need to manually load the tokenizer and model.

# ✔ Automatically handles tokenization

# It converts your text to tokens internally.

# ✔ Runs the model

# ✔ Returns clean, human-readable output

# do_sample=False

# This disables sampling and uses greedy decoding.

# 🔍 What is greedy decoding?

# The model picks the highest-probability next word at each step.

# Output is deterministic → same input → same summary every time.

# No randomness.

# If you set do_sample=True, the model would generate more diverse and creative summaries (but may vary each time).

# Colab paid products - Cancel contracts here

