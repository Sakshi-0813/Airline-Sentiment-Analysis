#lean and preprocess text using NLP

import re
import html
import nltk

from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet", quiet=True)

tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = html.unescape(str(text))
    text = text.lower()

    # Remove URLs and Twitter usernames
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    # Keep hashtag words, but remove the # symbol
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove remaining unwanted characters
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = tokenizer.tokenize(text)

    # Lemmatize without automatically removing stop words
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)

def preprocess_series(texts):
    if isinstance(texts, list):
        texts = pd.Series(texts)

    return texts.apply(preprocess_text)