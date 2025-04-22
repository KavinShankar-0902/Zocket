import streamlit as st
from newspaper import Article
from transformers import pipeline
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def fetch_article_text(link):
    try:
        news = Article(link)
        news.download()
        news.parse()
        return news.text
    except Exception as err:
        st.error(f"Error fetching article: {err}")
        return None

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_summary(raw_text, min_chars=30, max_chars=130):
    summarizer_pipeline = load_summarizer()
    if len(raw_text) > 1024:
        raw_text = raw_text[:1024]
    summary_output = summarizer_pipeline(
        raw_text,
        max_length=max_chars,
        min_length=min_chars,
        do_sample=False
    )
    return summary_output[0]['summary_text']


def main():
    st.title("AI Article Summarizer")
    st.write("This app fetches an article from a URL and generates a summary using AI.")


    article_url = st.text_input("Enter article URL:")
    min_length = st.slider("Minimum summary length (characters)", 30, 100, 30)
    max_length = st.slider("Maximum summary length (characters)", 100, 300, 130)

    if st.button("Summarize"):
        if article_url:
            with st.spinner("Fetching article..."):
                article_text = fetch_article_text(article_url)


            if article_text:
                with st.spinner("Generating summary..."):
                    summary_result = generate_summary(article_text, min_length, max_length)
                st.subheader("Summary")
                st.success(summary_result)

            else:
                st.error("Failed to process the URL.")
        else:
            st.warning("Please enter a URL first.")

if __name__ == "__main__":
    main()