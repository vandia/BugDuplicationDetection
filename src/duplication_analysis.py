import io_utilities as ut
#import reporter_reputation as rr
import cleaning_utilities as cl
import pandas as pd
#import similarity_calculation as sm

def main():
    #reputation = rr.generate_reputation()
    reputation = ut.load('../data_out/ReporterReputation.csv')
    cl.modify_column_types(reputation, {'active':int})
    #ut.save_csv(reputation, '../data_out/ReporterReputation.csv', False)
    #combined = sm.generate()
    combined = ut.load('../data_out/OscarBugSimilarities.csv')
    #ut.save_csv(combined, '../data_out/OscarBugSimilarities.csv', False)
    classified = ut.load('../data_in/OscarDuplicationClassification.csv')

    bsimcl = pd.merge(combined, classified, left_on=['bugid_1', 'bugid_2'], right_on=['bugid_1', 'bugid_2'], how='left')
    bsimcl.classifier.fillna("OTHER", inplace=True)
    cl.drop_columns(bsimcl, ['reporter'])
    ut.save_csv(bsimcl, '../data_out/OscarBugSimilaritiesClassified.csv', False)

    bsimrepcl = pd.merge(combined, reputation, left_on='reporter', right_on='lowerUserName', how='left')
    cl.drop_columns(bsimrepcl, ['lowerUserName','reporter','lowerEmailAddress','id'])
    bsimrepcl=cl.fill_nan_values(bsimrepcl,['active','reports_number','total_commits','seniority'])
    bsimrepcl = pd.merge(bsimrepcl, classified, left_on=['bugid_1', 'bugid_2'], right_on=['bugid_1', 'bugid_2'], how='left')
    bsimrepcl.classifier.fillna("OTHER", inplace=True)
    ut.save_csv(bsimrepcl, '../data_out/OscarBugSimilaritiesReporterReputationClassified.csv', False)




if __name__ == '__main__':
    main()
