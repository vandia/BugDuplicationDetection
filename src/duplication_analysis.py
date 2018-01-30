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
from sklearn.svm import SVC
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import itertools
from imblearn.over_sampling import SMOTE
from numpy.random import RandomState


def process_dataset():
    print("Generating the final dataset......")
    reputation = rr.generate_reputation()
    #reputation = ut.load('../data_out/ReporterReputation.csv')
    #cl.modify_column_types(reputation, {'active': int})
    ut.save_csv(reputation, '../data_out/ReporterReputation.csv', False)
    combined = sm.generate()
    # combined = ut.load('../data_out/OscarBugSimilarities.csv')
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
    bsimcl = process_dataset()
    #bsimcl = ut.load('../data_out/OscarBugSimilaritiesReporterReputationClassified.csv')

    print("Shape of complete dataset: ")
    print(bsimcl.shape)
    print("-------------")

    dfdet = ut.load('../data_in/OscarBugDetails.csv')
    dfdet_tr = dfdet[dfdet['status'].str.contains("Closed")]
    training_df = bsimcl[(bsimcl.bugid_1.isin(dfdet_tr.bugid)) & (bsimcl.bugid_2.isin(dfdet_tr.bugid))]
    predict_df = bsimcl[~((bsimcl.bugid_1.isin(dfdet_tr.bugid)) & (bsimcl.bugid_2.isin(dfdet_tr.bugid)))]
    results = predict_df.loc[:, ['bugid_1', 'bugid_2']]
    df_dup = training_df[training_df['classifier'] == 'DUPLICATED']
    training_df_dup = df_dup.sample(frac=0.5, random_state=RandomState(73))
    training_df = training_df[training_df['classifier'] == 'OTHER'].sample(n=training_df_dup.shape[0] * 500, random_state=RandomState(7))
    training_df = training_df.append([training_df_dup] * 150)
    test_df_dup = df_dup.sample(frac=0.5, random_state=RandomState(42))
    test_df = training_df[training_df['classifier'] == 'OTHER'].sample(n=test_df_dup.shape[0] * 100, random_state=RandomState(99))
    test_df = test_df.append([test_df_dup] * 25)

    print("Shape of training set")
    print(training_df.shape)
    print("-------------")
    print("Shape of testing set")
    print(test_df.shape)
    print("-------------")

    sm = SMOTE()
    scoring = ['accuracy', 'average_precision', 'precision', 'recall', 'f1']
    ytr_real = training_df['binary_classifier']
    x_basic, y_basic = sm.fit_sample(training_df[['categ_cosine_similarity', 'text_cosine_similarity']], ytr_real)
    x_report, y_report = sm.fit_sample(training_df[['categ_cosine_similarity', 'text_cosine_similarity', 'active',
                                                    'reports_number','total_commits', 'seniority']], ytr_real)
    X =[[x_basic, y_basic], [x_report, y_report]]

    ytst_real = test_df['binary_classifier']
    xtst_basic, ytst_basic = sm.fit_sample(test_df[['categ_cosine_similarity', 'text_cosine_similarity']], ytst_real)
    xtst_report, ytst_report = sm.fit_sample(test_df[['categ_cosine_similarity', 'text_cosine_similarity', 'active',
                                                      'reports_number','total_commits', 'seniority']], ytst_real)
    X_t = [[xtst_basic, ytst_basic],[xtst_report, ytst_report]]

    print("Shape of training set after SMOTE")
    print(len(X[0][0]))
    print(len(X[1][0]))
    print("-------------")
    print("Shape of testing set after SMOTE")
    print(len(X_t[0][0]))
    print(len(X_t[1][0]))
    print("-------------")

    X_p = [predict_df[['categ_cosine_similarity', 'text_cosine_similarity']],
           predict_df[['categ_cosine_similarity', 'text_cosine_similarity', 'active', 'reports_number',
                       'total_commits', 'seniority']]]

    labels = ['Decision Tree only bug information', 'Decision Tree with reputation analysis',
              'Naive Bayes only bug information', 'Naive Bayes with reputation analysis',
              'Random Forest only bug information', 'Random Forest with reputation analysis',
              'Extreme Randomized Tree only bug information', 'Extreme Randomized Tree with reputation analysis',
              'Adaboost Class. only bug information', 'Adaboost Class. with reputation analysis',
              'Logistic Regression only bug information', 'Logistic Regression with reputation analysis',
              'Support Vector Classification only bug information', 'Support Vector Classification with reputation analysis']

    CLF = [tree.DecisionTreeClassifier(class_weight='balanced'), BernoulliNB(),
           RandomForestClassifier(class_weight='balanced', n_estimators=50, verbose=1),
           ExtraTreesClassifier(verbose=1),
           AdaBoostClassifier(n_estimators=100), LogisticRegression(), SVC()]
    cv = ms.StratifiedShuffleSplit(n_splits=10, test_size=0.1)
    i = 0

    for clf in CLF:
        for x_tr, x_tst, x_pred in itertools.izip(X, X_t, X_p):
            print("-------------------------------------------------------------------------")
            print(labels[i])
            print("-------------------------------------------------------------------------")
            print ("Number of different class training dataset")
            print(np.bincount(x_tr[1]))
            print("---------------")
            print("Number of different class test dataset")
            print(np.bincount(x_tst[1]))
            print("---------------")
            scores = ms.cross_validate(clf, x_tr[0],x_tr[1], scoring=scoring, cv=cv)
            print("Training scores")
            print(scores)
            print("---------------")
            test = ms.cross_val_predict(clf, x_tst[0], x_tst[1], cv=10)

            #results['predicted_' + labels[i].replace(" ", "_")] = pd.Series(ms.cross_val_predict(estimator=clf, X=x_pred, y=None, cv=5))

            #Report
            score = metrics.roc_auc_score(x_tst[1], test, average='micro')
            print("Test roc auc score micro")
            print(score)
            print("------------")
            score = metrics.roc_auc_score(x_tst[1], test, average='macro')
            print("Test roc auc score macro")
            print(score)
            print("------------")
            score = metrics.accuracy_score(x_tst[1], test)
            print("Test accuracy score")
            print(score)
            print("------------")
            score = metrics.cohen_kappa_score(x_tst[1], test)
            print("Test cohen_kappa_score score")
            print(score)
            print("------------")
            score = metrics.precision_recall_fscore_support(x_tst[1], test, average='micro')
            print("Test precision_recall_fscore score micro")
            print(score)
            print("------------")
            score = metrics.precision_recall_fscore_support(x_tst[1], test, average='macro')
            print("Test precision_recall_fscore score macro")
            print(score)
            print("------------")
            plot_statistics(x_tst[1], test, labels[i])
            i += 1


if __name__ == '__main__':
    main()
