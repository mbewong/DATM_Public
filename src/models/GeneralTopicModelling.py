import tomotopy as tp
import pickle
# from gensim.test.utils import common_corpus, common_dictionary
# from gensim.models.ldamodel import LdaModel
# from gensim.models.coherencemodel import CoherenceModel


def getCorpus(input_data):
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(), stopwords=['.'])
    corpus.process(input_data)
    # corpus.process(open(input_file_non_noisy, encoding='utf-8'))
    #     print('### corpus processed')
    return corpus


def PTM_model(input_corpus, num_topics, a=None, b=None):
    if a is None:
        a = 1/num_topics
    if b is None:
        b = 1/num_topics
    #     print('...initialised')
    ptm_model = tp.PTModel(k=num_topics, alpha=a, eta=b, corpus=input_corpus)
    #     print('### PTM training initialised')
    for i in range(0, 1000, 10):
        ptm_model.train(10)
    #     print('### training completed')
    return ptm_model

# ptm_mdl = PTM_model(data_tweets)


def LDA_model(input_corpus, num_topics, a=None, b=None):
    if a is None:
        a = 1/num_topics
    if b is None:
        b = 1/num_topics
    #     print('...initialised')
    # corpus = getCorpus(input_corpus)
    lda_model = tp.LDAModel(k=num_topics, alpha=a, eta=b, corpus=input_corpus)
    #     print('### LDA training initialised')
    for i in range(0, 1000, 10):
        lda_model.train(10)
    # ptm_model.summary(topic_word_top_n=5)
    #     print('### LDA training completed')
    return lda_model


def get_topic_sets(path_set, k, textDes=None, mdl=None, val_set=None):
    topic_set = {}
    for key in path_set.keys():
        # print(key)
        with open(path_set[key], 'rb') as m:
            data = pickle.load(m)
            #           change 'text' to 'review' for maccas
            if textDes is not None:
                cor = (key in ['noisy', 'non_noisy'] and getCorpus(data[textDes]) or getCorpus(data))
            else:
                cor = getCorpus(data)
        #         cor =  (key in ['noisy', 'non_noisy'] and 'review' or 'no review' )
        #         print(cor)
        if val_set is None:
            #         !!! IMPORTANT, WE HAVE TO SET ALPHA AND BETA MANUALLY
            if mdl is None or mdl == 'lda':
                mdl_lda = LDA_model(cor, k)
            else:
                mdl_lda = LDA_model(cor, k)
            topics = get_mdl_topics(mdl_lda, k)
            topic_set[key] = topics
        elif len(val_set) == 1:
            #         !!! IMPORTANT, WE HAVE TO SET ALPHA AND BETA MANUALLY
            if mdl is None or mdl == 'lda':
                mdl_lda = LDA_model(cor, k, a=val_set[0], b=val_set[0])
            else:
                mdl_lda = LDA_model(cor, k, a=val_set[0], b=val_set[0])
            topics = get_mdl_topics(mdl_lda, k)
            topic_set[key] = topics
        elif val_set is not None:
            topic_dict = {}
            for b in val_set:
                if mdl is None or mdl == 'lda':
                    mdlLDA = LDA_model(cor, k, a=b, b=b)
                else:
                    mdlLDA = LDA_model(cor, k, a=b, b=b)
                topics = get_mdl_topics(mdlLDA, k)
                topic_dict[b] = topics
            topic_set[key] = topic_dict
        # print(key)
    return topic_set


def get_mdl_topics(mdl, k):
    k_topics = []
    for k in range(k):
        word_set = mdl.get_topic_words(k, top_n=10)
        k_topics.append([w[0]for w in word_set])
    return k_topics

