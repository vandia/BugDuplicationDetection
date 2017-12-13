import io_utilities as ut
import reporter_reputation as rr
import cleaning_utilities as cl
import pandas as pd
import similarity_calculation as sm
from sklearn import svm

def process_dataset():
    reputation = rr.generate_reputation()
    #reputation = ut.load('../data_out/ReporterReputation.csv')
    cl.modify_column_types(reputation, {'active':int})
    ut.save_csv(reputation, '../data_out/ReporterReputation.csv', False)
    combined = sm.generate()
    #combined = ut.load('../data_out/OscarBugSimilarities.csv')
    ut.save_csv(combined, '../data_out/OscarBugSimilarities.csv', False)
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
    cl.drop_columns(bsimcl, ['reporter','classifier_second'])
    bsimcl.classifier.fillna("OTHER", inplace=True)
    bsimcl['binary_classifier']= bsimcl['classifier'].astype('category').cat.codes
    ut.save_csv(bsimcl, '../data_out/OscarBugSimilaritiesClassified.csv', False)

    bsimrepcl = pd.merge(combined, reputation, left_on='reporter', right_on='lowerUserName', how='left')
    cl.drop_columns(bsimrepcl, ['lowerUserName','reporter','lowerEmailAddress','id'])
    bsimrepcl=cl.fill_nan_values(bsimrepcl,['active','reports_number','total_commits','seniority'])
    bsimrepcl = pd.merge(bsimrepcl, classified, left_on=['bugid_1', 'bugid_2'], right_on=['bugid_1', 'bugid_2'],
                         how='left',suffixes=['','_first'])
    bsimrepcl = pd.merge(bsimrepcl, classified_rev, left_on=['bugid_1', 'bugid_2'], right_on=['bugid_1', 'bugid_2'],
                      how='left',suffixes=['','_second'])
    # bsimrepcl['classifier']=bsimrepcl.apply( lambda row: row['classifier'] if pd.isnull(row['classifier_second'])
    #                                     else row['classifier_second'], axis=1)
    bsimrepcl['classifier'] = pd.concat(
        [bsimrepcl['classifier'].dropna(), bsimrepcl['classifier_second'].dropna()]).reindex_like(bsimrepcl)

    cl.drop_columns(bsimrepcl, ['classifier_second'])
    bsimrepcl.classifier.fillna("OTHER", inplace=True)
    bsimrepcl['binary_classifier'] = bsimrepcl['classifier'].astype('category').cat.codes

    ut.save_csv(bsimrepcl, '../data_out/OscarBugSimilaritiesReporterReputationClassified.csv', False)


def main():
    process_dataset()
    bsimcl = ut.load('../data_out/OscarBugSimilaritiesClassified.csv')
    bsimrepcl = ut.load('../data_out/OscarBugSimilaritiesReporterReputationClassified.csv')





if __name__ == '__main__':
    main()
