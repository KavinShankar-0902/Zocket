# 📰 AI Article Summarizer

A simple yet powerful **Streamlit app** that lets you extract and summarize articles directly from a URL using **state-of-the-art NLP models**.

Built using:
- 💬 [Hugging Face Transformers](https://huggingface.co/transformers/)
- 🖥️ [Streamlit](https://streamlit.io/)
- 🗞️ [Newspaper3k](https://github.com/codelucas/newspaper)

---

## ✨ Features

- 🌐 Fetch full news articles from any URL
- 🤖 Summarize using `distilbart-cnn-12-6`, a distilled version of Facebook's BART
- 🔧 Adjustable minimum & maximum summary length
- ⚡ Fast inference and real-time results
- 🧠 Caches the model for better performance


## 📦 Installation

> Recommended: Use a virtual environment

```bash
# 1. Clone the repo (or copy the script)
git clone https://github.com/yourusername/article-summarizer.git
cd article-summarizer

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
