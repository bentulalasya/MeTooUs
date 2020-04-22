import re
import gensim
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer


# LDA model. Get topic proportions
def compute_coherence_values(corpus, processed_docs, dictionary, k, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=5000,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    main_df = pd.DataFrame()

    for cur_corpus in corpus:
        cur_proportion = lda_model.get_document_topics(cur_corpus)
        df = pd.DataFrame.from_records(cur_proportion)
        df_t = df.T
        df_without = df_t.iloc[1:]
        main_df = main_df.append(df_without)


    main_df.reset_index(drop=True, inplace=True)
    processed_docs.reset_index(drop=True, inplace=True)

    concated_df = pd.concat([processed_docs, main_df], axis=1)
    concated_df = concated_df.drop(['New_Contents'], axis=1)

    concated_df.to_csv('Tweet_Data_With_Topic_Proportion_' + str(k) + '.csv', mode='a', header=True, index=False)
    return lda_model


#Read tweet data and preprosess
def read_tweet_data(file_path):
    file_data = pd.read_csv(file_path)
    file_data["New_Contents"] = file_data["Contents"].apply(preprocesssing)
    file_data.drop(columns=['Contents'])
    return file_data


def preprocesssing(x):
    x = re.sub(r'http\S+', '', x)
    x = re.sub(r'@\S+', '', x)
    x = re.sub('[,\.!?]', '', x)
    x = x.lower()
    x = re.sub(r'metoo\S+', '', x)
    x = re.sub(r'meto\S+', '', x)
    x = re.sub(r'women\S+', '', x)
    x = re.sub(r'men\S+', '', x)
    x = re.sub(r'man\S+', '', x)
    return x.lower()


def secondary_preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def calculate_topic_proportion(filepath):
    np.random.seed(2018)

    #Change this to where the tweets data reside
    tweet_data = read_tweet_data(filepath)

    tweet_data["Preprocessed_Text"] = tweet_data['New_Contents'].map(secondary_preprocess)

    # Create Dictionary
    dictionary = gensim.corpora.Dictionary(tweet_data["Preprocessed_Text"])

    # Term Document Frequency
    corpus = [dictionary.doc2bow(doc) for doc in tweet_data["Preprocessed_Text"]]

    # Validation sets
    corpus_sets = [corpus]

    # get the coherence score for the given parameters
    lda_model = compute_coherence_values(corpus=corpus_sets[0], processed_docs=tweet_data,
                                         dictionary=dictionary,
                                         k=15, a=0.31, b=0.91)

# Calulate topic proportion for given input data.
calculate_topic_proportion("Tweet_Data_With_Cols.csv")