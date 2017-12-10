import io_utilities as ut
import reporter_reputation as rr
import cleaning_utilities as cl
import pandas as pd
import similarity_calculation as sm

def main():
    reputation = rr.generate_reputation()
    # reputation = ut.load('../data_out/ReporterReputation.csv')
    ut.save_csv(reputation, '../data_out/ReporterReputation.csv', False)
    combined = sm.generate()
    #combined = ut.load('../data_out/OscarBugSimilarities.csv')
    #ut.save_csv(combined, '../data_out/OscarBugSimilarities.csv', False)
    classified = ut.load('../data_in/OscarDuplicationClassification.csv')

    result = pd.merge(combined, classified, left_on=['bugid_1', 'bugid_2'], right_on=['bugid_1', 'bugid_2'], how='left')
    result.classifier.fillna("OTHER", inplace=True)
    ut.save_csv(result, '../data_out/OscarBugSimilaritiesClassified.csv', False)

    result = pd.merge(combined, reputation, left_on='reporter', right_on='lowerUserName', how='left')
    cl.drop_columns(result, ['lowerUserName'])
    result = pd.merge(result, classified, left_on=['bugid_1', 'bugid_2'], right_on=['bugid_1', 'bugid_2'], how='left')
    ut.save_csv(result, '../data_out/OscarBugSimilaritiesReporterReputationClassified.csv', False)


if __name__ == '__main__':
    main()
