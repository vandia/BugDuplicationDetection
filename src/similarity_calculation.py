import itertools as it
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

def clean_string(str):
    for i in ['.', '"', ',', '(', ')', '!', '?', ';', ':', '=', '*', '-', '\\', '/', '>']:
        str = str.replace(i, '')
    return str



def generate_doc2vec(df):

    print("Generating and training doc2vec model")

    df['comments'] = df['comments'].apply(lambda x: x.decode("utf-8"))
    df['comments'] = df['comments'].apply(lambda x: x.rstrip())

    df['tokenized'] = df['comments'].apply(lambda x: [w for w in gensim.utils.simple_preprocess(x, deacc=True)
                                                      if not w in nltk.corpus.stopwords.words('english')])

    df['taggedDocs'] = df.apply(lambda row: gensim.models.doc2vec.TaggedDocument(row['tokenized'], [row['bugid']]),
                                axis=1)

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

    print("Generating contextual and text similarities")

    result = pd.DataFrame(columns=['bugid_1', 'bugid_2', 'categ_cosine_similarity', 'text_cosine_similarity'])
    bugid_1 = pd.Series()
    bugid_2 = pd.Series()
    categ_cosine_similarity = pd.Series()
    text_cosine_similarity = pd.Series()

    pairs = generate_pairs(details)
    crop_details = preprocess_categorical(details)

    for row in pairs:
        grt = crop_details.loc[row[1]]
        if grt.size == 0:
            continue
        sim_categorical = cosine_similarity(crop_details.loc[[row[0]]], Y=grt)
        sim_textual = cosine_similarity(comments.loc[[row[0]]].vector.tolist(), Y=comments.loc[row[1]].vector.tolist())
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


def preprocess_categorical(dataset):

    print("Preprocessing categorical values from bug")

    df = dataset.copy(deep=True)
    df['project'] = df['project'].astype('category').cat.codes
    df['reporter'] = df['reporter'].astype('category').cat.codes
    df['status'] = df['status'].astype('category').cat.codes
    df['sourceforge_reporter'] = df['sourceforge_reporter'].astype('category').cat.codes
    df = cl.drop_columns(df, ['classifier', 'creation_date'])
    df.set_index('bugid', inplace=True)

    return df


def generate():
    comments = ut.load('../data_in/OscarBugComments.csv')
    comments = generate_doc2vec(comments)
    # print (comments)
    details = ut.load('../data_in/OscarBugDetails.csv', date_cols=['creation_date'])
    combined = generate_similarity(details, comments)

    return combined
