import io_utilities as ut
import numpy as np
#import reporter_reputation as rr
import cleaning_utilities as cl
import pandas as pd
#import similarity_calculation as sm
from sklearn import metrics
from sklearn import svm
import sklearn.model_selection as ms
import matplotlib.pyplot as plt

def process_dataset():
    #reputation = rr.generate_reputation()
    reputation = ut.load('../data_out/ReporterReputation.csv')
    cl.modify_column_types(reputation, {'active':int})
    #ut.save_csv(reputation, '../data_out/ReporterReputation.csv', False)
    #combined = sm.generate()
    combined = ut.load('../data_out/OscarBugSimilarities.csv')
    #ut.save_csv(combined, '../data_out/OscarBugSimilarities.csv', False)
    classified = ut.load('../data_in/OscarDuplicationClassification.csv')
    classified_rev=classified.copy(deep=True)
    classified_rev=classified_rev.rename(columns = {'bugid_2':'bugid_1_bis', 'bugid_1':'bugid_2'})
    classified_rev = classified_rev.rename(columns={'bugid_1_bis': 'bugid_1'})

    bsimcl = pd.merge(combined, classified, left_on=['bugid_1', 'bugid_2'], right_on=['bugid_1', 'bugid_2'], how='left',
                      suffixes=['','_first'])
    bsimcl = pd.merge(bsimcl, classified_rev, left_on=['bugid_1', 'bugid_2'], right_on=['bugid_1', 'bugid_2'], how='left',
                      suffixes=['', '_second'])
    # bsimcl['classifier']=bsimcl.apply( lambda row: row['classifier'] if pd.isnull(row['classifier_second'])
    #                                     else row['classifier_second'], axis=1)
    bsimcl['classifier'] = pd.concat([bsimcl['classifier'].dropna(), bsimcl['classifier_second'].dropna()]).reindex_like(bsimcl)

    bsimcl = pd.merge(bsimcl, reputation, left_on='reporter', right_on='lowerUserName', how='left')
    bsimcl = cl.fill_nan_values(bsimcl, ['active', 'reports_number', 'total_commits', 'seniority'])

    cl.drop_columns(bsimcl, ['classifier_second','classifier_second','lowerUserName','reporter','lowerEmailAddress','id',
                             'lowerEmailAddress','lowerUserName'])
    bsimcl.classifier.fillna("OTHER", inplace=True)
    bsimcl['binary_classifier']= bsimcl['classifier'].astype('category').cat.codes

    ut.save_csv(bsimcl, '../data_out/OscarBugSimilaritiesClassified.csv', False)
    ut.save_csv(bsimcl, '../data_out/OscarBugSimilaritiesReporterReputationClassified.csv', False)

    return bsimcl

def plot_statistics(y_real,y_pred, label=None):

    fpr, tpr, thresholds = ms.roc_curve(y_real, y_pred)
    roc_auc = ms.auc(fpr, tpr)
    f = plt.figure()
    ax = f.add_subplot(111)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, ax=ax,
             label='ROC  (AUC = %0.2f)' % (roc_auc))
    plt.savefig("../data_out/Completion Event percentage histogram.svg", dpi=300)

def main():

    #bsimcl = process_dataset()
    bsimcl = ut.load('../data_out/OscarBugSimilaritiesReporterReputationClassified.csv')
    scoring = ['accuracy','average_precision', 'precision', 'recall', 'f1']
    X=[np.asarray(bsimcl[['categ_cosine_similarity','text_cosine_similarity', 'active', 'reports_number',
                'total_commits', 'seniority']]), np.asarray(bsimcl[['categ_cosine_similarity','text_cosine_similarity']])]
    y_real = np.asarray(bsimcl['binary_classifier'])

    for x in X:

        clf = svm.SVC()
        cv=ms.StratifiedShuffleSplit(n_splits=100, test_size=0.1)
        scores = ms.cross_validate(clf, x, y_real, scoring=scoring, cv=cv)
        print(scores)
        predicted = ms.cross_val_predict(clf, x, y_real, cv=10)
        score = metrics.accuracy_score(y_real, predicted)
        plot_statistics(y_real,predicted)
        print(score)


if __name__ == '__main__':
    main()
