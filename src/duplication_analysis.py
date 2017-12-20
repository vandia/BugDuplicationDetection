import io_utilities as ut
import numpy as np
import reporter_reputation as rr
import cleaning_utilities as cl
import pandas as pd
import similarity_calculation as sm
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import itertools


def process_dataset():
    print("Generating the final dataset")
    reputation = rr.generate_reputation()
    #reputation = ut.load('../data_out/ReporterReputation.csv')
    #cl.modify_column_types(reputation, {'active': int})
    ut.save_csv(reputation, '../data_out/ReporterReputation.csv', False)
    combined = sm.generate()
    #combined = ut.load('../data_out/OscarBugSimilarities.csv')
    ut.save_csv(combined, '../data_out/OscarBugSimilarities.csv', False)
    classified = ut.load('../data_in/OscarDuplicationClassification.csv')
    classified_rev = classified.copy(deep=True)
    classified_rev = classified_rev.rename(columns={'bugid_2': 'bugid_1_bis', 'bugid_1': 'bugid_2'})
    classified_rev = classified_rev.rename(columns={'bugid_1_bis': 'bugid_1'})

    bsimcl = pd.merge(combined, classified, left_on=['bugid_1', 'bugid_2'], right_on=['bugid_1', 'bugid_2'], how='left',
                      suffixes=['', '_first'])
    bsimcl = pd.merge(bsimcl, classified_rev, left_on=['bugid_1', 'bugid_2'], right_on=['bugid_1', 'bugid_2'],
                      how='left',
                      suffixes=['', '_second'])
    bsimcl['classifier'] = pd.concat(
        [bsimcl['classifier'].dropna(), bsimcl['classifier_second'].dropna()]).reindex_like(bsimcl)

    bsimcl = pd.merge(bsimcl, reputation, left_on='reporter', right_on='lowerUserName', how='left')

    cl.drop_columns(bsimcl,
                    ['classifier_second', 'classifier_second', 'lowerUserName', 'reporter', 'lowerEmailAddress', 'id',
                     'lowerEmailAddress', 'lowerUserName'])
    bsimcl.classifier.fillna("OTHER", inplace=True)
    bsimcl = cl.fill_nan_values(bsimcl, ['active', 'reports_number', 'total_commits', 'seniority'])
    bsimcl['binary_classifier'] = bsimcl['classifier'].astype('category').cat.codes

    ut.save_csv(bsimcl, '../data_out/OscarBugSimilaritiesReporterReputationClassified.csv', False)

    return bsimcl


def plot_statistics(y_real, y_pred, label):
    print("Plotting " + label)

    fpr, tpr, thresholds = metrics.roc_curve(y_real, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    l = 'ROC (AUC = ' + str(roc_auc) + ')'
    plt.figure()
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label=l, color='red')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(label)
    plt.legend(loc="lower right")
    plt.savefig("../data_out/" + label + ".png", dpi=300)


def main():
    # bsimcl = process_dataset()
    bsimcl = ut.load('../data_out/OscarBugSimilaritiesReporterReputationClassified.csv')
    rep_row = bsimcl[bsimcl['classifier'] == 'DUPLICATED']
    print("Number of duplicated rows")
    print(rep_row.shape)
    print("-------------")
    dfdet = ut.load('../data_in/OscarBugDetails.csv')
    dfdet_tr = dfdet[dfdet['status'].str.contains("Closed")]
    # merged = dfdet_tr.merge(dfdet_tr, indicator=True, how='outer')
    # dfdet_tst = merged[merged['_merge'] == 'left_only']
    # print (dfdet_tst)
    print("Shape of complete dataset: ")
    print(bsimcl.shape)
    print("-------------")

    training_df = bsimcl[(bsimcl.bugid_1.isin(dfdet_tr.bugid)) & (bsimcl.bugid_2.isin(dfdet_tr.bugid))]
    test_df = bsimcl[~((bsimcl.bugid_1.isin(dfdet_tr.bugid)) & (bsimcl.bugid_2.isin(dfdet_tr.bugid)))]
    results = test_df.loc[:, ['bugid_1', 'bugid_2']]
    print("Shape of training dataset without padding: ")
    print(training_df.shape)
    print("-------------")
    print("Shape of test dataset without padding: ")
    print(test_df.shape)
    print("-------------")
    # training_df = training_df.append([rep_row] * (training_df.shape[0] / 1000))
    # training_df = training_df.sample(frac=1).reset_index(drop=True)
    # print("Shape of training dataset after padding: " )
    # print(training_df.shape)
    # print("-------------")
    # test_df = test_df.append([rep_row] * (test_df.shape[0] / 1000))
    # print("Shape of test dataset after padding: ")
    # print(test_df.shape)
    # print("-------------")

    scoring = ['accuracy', 'average_precision', 'precision', 'recall', 'f1']
    X = [training_df[['categ_cosine_similarity', 'text_cosine_similarity']],
         training_df[['categ_cosine_similarity', 'text_cosine_similarity', 'active', 'reports_number',
                      'total_commits', 'seniority']]]
    y_real = training_df['binary_classifier']

    X_t = [test_df[['categ_cosine_similarity', 'text_cosine_similarity']],
           test_df[['categ_cosine_similarity', 'text_cosine_similarity', 'active', 'reports_number',
                    'total_commits', 'seniority']]]
    y_test_real = test_df['binary_classifier']
    labels = ['Naive Bayes only bug information', 'Naive Bayes with reputation analysis',
              'Random Forest only bug information', 'Random Forest with reputation analysis',
              'Extreme Randomized Tree only bug information', 'Extreme Randomized Tree with reputation analysis',
              'Adaboost Class. only bug information', 'Adaboost Class. with reputation analysis',
              'Logistic Regression only bug information', 'Logistic Regression with reputation analysis']
    CLF = [BernoulliNB(), RandomForestClassifier(class_weight='balanced', n_estimators=50, verbose=1),
           ExtraTreesClassifier(verbose=1),
           AdaBoostClassifier(n_estimators=100), LogisticRegression()]
    cv = ms.StratifiedShuffleSplit(n_splits=10, test_size=0.1)
    i = 0

    for clf in CLF:
        for x_tr, x_tst in itertools.izip(X, X_t):
            print(labels[i])
            scores = ms.cross_validate(clf, x_tr, y_real, scoring=scoring, cv=cv)
            print("Training scores")
            print(scores)
            print("---------------")
            predicted = ms.cross_val_predict(clf, x_tr, y_real, cv=5)
            results['predicted_' + labels[i].replace(" ", "_")] = pd.Series(
                ms.cross_val_predict(clf, x_tst, y_test_real, cv=5))
            score = metrics.roc_auc_score(y_real, predicted, average='micro')
            print("Test roc auc score")
            print(score)
            print("------------")
            score = metrics.accuracy_score(y_real, predicted)
            print("Test accuracy score")
            print(score)
            print("------------")
            score = metrics.cohen_kappa_score(y_real, predicted)
            print("Test cohen_kappa_score score")
            print(score)
            print("------------")
            score = metrics.precision_recall_fscore_support(y_real, predicted, average='micro')
            print("Test precision_recall_fscore score")
            print(score)
            print("------------")
            plot_statistics(y_real, predicted, labels[i])
            i += 1


if __name__ == '__main__':
    main()
