import tqdm
from gensim import corpora
from gensim.models import CoherenceModel
import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
import re
import pyLDAvis.gensim
import pickle
import pyLDAvis


# Visualize the topics
# pyLDAvis.enable_notebook()
# supporting function
def compute_coherence_values(corpus, processed_docs, dictionary, k, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=5000,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    print("k = ")
    print(k)
    print(a)
    print(b)
    main_df = pd.DataFrame()
    for cur_corpus in corpus:
        cur_proportion = lda_model.get_document_topics(cur_corpus)
        df = pd.DataFrame.from_records(cur_proportion)
        df_t = df.T
        df_without = df_t.iloc[1:]
        main_df = main_df.append(df_without)

    main_df['Date'] = processed_docs['Date (GMT)']

    main_df.to_csv('test'+str(k)+'.csv', mode='a', header=True, index=False)


    #visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    #pyLDAvis.save_html(visualisation, 'LDA_Visualization_' + str(k) + ".html")

    # coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')

    return lda_model


grid = {}
grid['Validation_Set'] = {}
# Topics range
min_topics = 10
max_topics = 40
step_size = 5
topics_range = range(min_topics, max_topics, step_size)
# Alpha parameter
alpha = list(np.arange(0.31, 1, 1.3))

# Beta parameter
beta = list(np.arange(0.91, 1, 1.3))
print("Before")
# Data

print("After 1")
np.random.seed(2018)
stemmer = SnowballStemmer('english')


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
    x = re.sub('metoo', '', x)
    x = re.sub('meto', '', x)
    return x.lower()


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


processed_docs = read_tweet_data('./../Output/Tweet_Data_With_Dates.csv')

print("After 1")
np.random.seed(2018)
stemmer = SnowballStemmer('english')

processed_docs["Preprocessed_Text"] = processed_docs['New_Contents'].map(preprocess)
# Create Dictionary
dictionary = gensim.corpora.Dictionary(processed_docs["Preprocessed_Text"])

# Term Document Frequency
corpus = [dictionary.doc2bow(doc) for doc in processed_docs["Preprocessed_Text"]]
print("After 3")

# Validation sets
num_of_docs = len(corpus)
corpus_sets = [corpus]
corpus_title = ['75% Corpus', '100% Corpus']
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                 }

# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=540)

    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    lda_model = compute_coherence_values(corpus=corpus_sets[i], processed_docs=processed_docs,
                                                  dictionary=dictionary,
                                                  k=k, a=a, b=b)


                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    #model_results['Coherence'].append(cv)

                    pbar.update(1)

    pbar.close()

# 10	0.31	0.91	0.580626898
