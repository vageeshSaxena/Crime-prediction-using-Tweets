from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from utils.consts import LDA_PARAMS


def coalesce(token):
    """
    Klaues: why this function?
    """

    new_tokens = []
    for char in token:
        if len(new_tokens) < 2 or char != new_tokens[-1] or char != new_tokens[-2]:
            new_tokens.append(char)
    return ''.join(new_tokens)


def preprocess_tweet_for_LDA(raw_tokens):
    """
    text input is one string
    output is tokenized and preprocessed(as defined below) text

    lowercase
    no hashtags or mentions
    any url converted to "url"
    replace multiple repeated chars with 2 of them. eg paaaarty -> paarty
    """

    processed_tokens = []
    for token in raw_tokens:
        if token.startswith("@") or token.startswith("#"):
            continue
        elif token.startswith("https://") or token.startswith("http://"):
            processed_tokens.append("url")
        else:
            processed_tokens.append(coalesce(token))

    return processed_tokens


def train_LDA_model(docs, params=LDA_PARAMS, preprocessor=preprocess_tweet_for_LDA):

    vectorizer = CountVectorizer(stop_words="english",
                                 preprocessor=preprocessor,
                                 tokenizer=lambda x: x)

    lda_train_data = vectorizer.fit_transform(docs)

    lda_model = LatentDirichletAllocation(**params)

    lda_model.fit(lda_train_data)

    doc_topics = lda_model.transform(lda_train_data)

    vocabulary = vectorizer.get_feature_names()

    return lda_model, doc_topics, vocabulary


def get_topic_top_words_LDA(topic_index, lda_model, vocabulary, n_top_words):
    """
    Return the top words for given topic.
    """

    topic = lda_model.components_[topic_index]
    return [vocabulary[i] for i in topic.argsort()[:-n_top_words - 1:-1]]


def print_top_words_LDA(lda_model, vocabulary, n_top_words, non_trival=True):
    """
    Print the top words for each topic.
    with `non_trivial` set to True, only topics without uniform distribution are shown.
    """

    for topic_index in range(len(lda_model.components_)):
        if not (non_trival and len(set(lda_model.components_[topic_index])) == 1):

            message = "Topic #%d: " % topic_index
            message += " | ".join(get_topic_top_words_LDA(topic_index,
                                                          lda_model, vocabulary, n_top_words))
            print(message)

    print()
