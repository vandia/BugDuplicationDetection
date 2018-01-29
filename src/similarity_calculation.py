import itertools as it
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import cleaning_utilities as cl
import io_utilities as ut
import gensim
import nltk


nltk.download('punkt')
nltk.download('stopwords')


# it needs a dataset with all the vector values for each bug ('dataset') and
# another dataset('combined') the possible combination of two bugs (bugid_1, bugid_2)


def generate_doc2vec(df):
    print("Generating and training doc2vec model")

    df['comments'] = df['comments'].apply(lambda x: x.decode("utf-8"))
    df['comments'] = df['comments'].apply(lambda x: x.rstrip())

    df['tokenized'] = df['comments'].apply(lambda x: [w for w in gensim.utils.simple_preprocess(x, deacc=True)
                                                      if not w in nltk.corpus.stopwords.words('english')])

    df['taggedDocs'] = df.apply(lambda row: gensim.models.doc2vec.TaggedDocument(row['tokenized'], [row['bugid']]), axis=1)

    model = gensim.models.doc2vec.Doc2Vec(size=400, min_count=2, iter=55)
    model.build_vocab(df['taggedDocs'])
    model.train(df['taggedDocs'], total_examples=model.corpus_count, epochs=model.iter)

    df['vector'] = df['tokenized'].apply(lambda x: model.infer_vector(x))
    df.set_index('bugid', inplace=True)

    return df


# generate all the possible combinations using A priori algorithm sorting by date.
def generate_pairs(dataset):
    print("Generating bug pair combinations")

    result = []
    dataset.sort_values('creation_date', ascending=True, kind='mergesort', inplace=True)
    for index, row in dataset.iterrows():
        grt = dataset.loc[dataset['creation_date'] > row['creation_date']]
        if (grt.size == 0):
            continue
        result.append([row['bugid'], grt['bugid'].tolist()])
    return result


# generate context and text similarities.
def generate_similarity(details, comments):
    result = pd.DataFrame(columns=['bugid_1', 'bugid_2', 'categ_cosine_similarity', 'text_cosine_similarity'])
    bugid_1 = pd.Series()
    bugid_2 = pd.Series()
    categ_cosine_similarity = pd.Series()
    text_cosine_similarity = pd.Series()

    pairs = generate_pairs(details)
    det_indexed= details[['bugid','vector']].set_index('bugid')

    print("Generating contextual and text similarities")

    for row in pairs:
        grt = det_indexed.loc[row[1]].vector
        if grt.size == 0:
            continue
        sim_categorical = cosine_similarity(det_indexed.loc[[row[0]]].vector.tolist(), Y=grt.tolist())
        if (sim_categorical.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(sim_categorical.sum())
                and not np.isfinite(sim_categorical).all()):
            print("ERROR")
            print(row)
            print(np.where(np.isinf(sim_categorical)))

        sim_textual = cosine_similarity(comments.loc[[row[0]]].vector.tolist(), Y=comments.loc[row[1]].vector.tolist())
        if (sim_textual.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(sim_textual.sum())
                and not np.isfinite(sim_textual).all()):
            print("ERROR")
            print(row)
            print(np.where(np.isinf(sim_categorical)))
        bugid_2 = bugid_2.append(pd.Series(row[1]))
        bugid_1 = bugid_1.append(pd.Series(list(it.repeat(row[0], len(row[1])))))
        categ_cosine_similarity = categ_cosine_similarity.append(pd.Series(sim_categorical[0, :]))
        text_cosine_similarity = text_cosine_similarity.append(pd.Series(sim_textual[0, :]))

    result['bugid_1'] = bugid_1
    result['bugid_2'] = bugid_2
    result['categ_cosine_similarity'] = categ_cosine_similarity
    result['text_cosine_similarity'] = text_cosine_similarity
    result['text_cosine_similarity'] = text_cosine_similarity
    result['text_cosine_similarity'] = text_cosine_similarity
    result = pd.merge(result, details[['bugid', 'reporter']], left_on='bugid_2', right_on='bugid', how='left')
    result = cl.drop_columns(result, ['bugid'])
    return result


def preprocess_categorical(df):
    print("Preprocessing categorical values from bug")

    df['concatenated'] = df['type'].astype(str) + ' ' + df['priority'].astype(str) + ' ' + df['workflowId'].astype(
        str) + ' ' + df['version'].astype(str)
    df['concatenated'] = df['concatenated'].apply(lambda x: x.decode("utf-8"))
    df['concatenated'] = df['concatenated'].apply(lambda x: x.rstrip())
    df['tokenized'] = df['concatenated'].apply(lambda x: nltk.word_tokenize(x))
    df['taggedDocs']= df[['tokenized','bugid']].apply(lambda r: gensim.models.doc2vec.TaggedDocument(
        r['tokenized'],[r['bugid']]), axis='columns')
    model = gensim.models.doc2vec.Doc2Vec(size=400, min_count=1, iter=55)
    model.build_vocab(df['taggedDocs'])
    model.train(df['taggedDocs'], total_examples=model.corpus_count, epochs=model.iter)
    df['vector'] = df['tokenized'].apply(lambda x: model.infer_vector(x))
    df = cl.drop_columns(df, ['concatenated', 'tokenized', 'taggedDocs'])
    return df


def generate():
    comments = ut.load('../data_in/OscarBugComments.csv')
    comments = generate_doc2vec(comments)
    # print (comments)

    details = ut.load('../data_in/OscarBugDetails.csv', date_cols=['creation_date'])
    details = preprocess_categorical(details)
    combined = generate_similarity(details, comments)



    return combined
