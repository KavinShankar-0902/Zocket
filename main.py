import streamlit as st
from newspaper import Article
from transformers import pipeline

def get_url_text(link):
    try:
        news = Article(link)
        news.download()
        news.parse()
        return news.title, news.text

    except Exception as err:
        st.error(f"Error fetching article: {err}")
        return None, None


@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device = -1)


def summary(raw_text, min_chars, max_chars):
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


def extract_key_info(title, content, min_length, max_length):
    s = summary(content, min_chars=min_length, max_chars=max_length)
    return s


def main():
    st.title("📰 AI Article Summarizer using Web Scraping")

    article_url = st.text_input("Enter article URL:")
    min_length = st.slider("Min summary length", 30, 500, 30)
    max_length = st.slider("Max summary length", 100, 900, 130)

    if st.button("Summarize"):
        if article_url:
            with st.spinner("Fetching URL content..."):
                title, article_text =  get_url_text(article_url)

            if article_text:
                with st.spinner("Generating summary..."):

                    summary_result = extract_key_info(title, article_text, min_length, max_length)
                st.subheader("📝 Key Information")
                st.success(summary_result)
            else:
                st.error("Failed to process the URL.")
        else:
            st.warning("Please enter a URL first.")


if __name__ == "__main__":
    main()
