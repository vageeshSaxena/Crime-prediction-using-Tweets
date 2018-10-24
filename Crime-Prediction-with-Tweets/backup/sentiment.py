
def find_sentiment_doc(list_Sentence):
    """input :
        list of sentences"""
    # return data.predict_proba(list_Sentence)
    return data.score_document(list_Sentence)

# For finding the sentiment of some particular documents


def sentiment_of_document(documents):
    """input :
        a) List of List of List
        b) tokenized words"""
    sentiment_list, tweets_list, document_list = ([] for i in range(3))
    for document in documents:
        for tweets in document:
            tweets_list.append(" ".join(tweets))
        document_list.append(tweets_list)
        sentiment_list.append(find_sentiment_doc(document_list[-1]))
    return sentiment_list


df = pd.DataFrame(columns=['Topics', 'Results'])
df["Topics"] = ["Number of tweets in Corpus", "Average micro test precision percentage",
                "Average macro test precision percentage", "Average micro test recall percentage", "Average macro test recall percentage"]
df["Results"] = [str(len(corpus)), "75.5490 %", "75.5868 %", "75.5490 %", "75.5490 %"]
print(df)
