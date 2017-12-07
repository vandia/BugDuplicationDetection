import io_utilities as ut
import cleaning_utilities as cl
import pandas as pd
#import SimilarityCalculation as sm


def generate_reputation():
    # Creation of reporter social parameters

    due = ut.load('../data_in/OscarUserExperience.csv', date_cols=['createdDate', 'updatedDate'])
    dgit = ut.load('../data_in/OscarGitLog.csv', date_cols=['date'])
    djira = ut.load('../data_in/OscarBugDetails.csv')

    due = cl.drop_columns(due, ['directoryId', 'updatedDate', 'displayName', 'lowerDisplayName',
                                'emailAddress', 'credential', 'localServiceDeskUser'])

    dgit['author_email'] = dgit.author_email.str.lower()
    dgb = dgit.groupby('author_email').size().reset_index(name='commits')
    djira = djira.groupby('reporter').size().reset_index(name='reports_number')

    ##############################################################
    # merge between the user experience in JIRA and git log info #
    ##############################################################
    result = pd.merge(due, dgb, left_on='lowerEmailAddress', right_on='author_email', how='left')
    result = pd.merge(result, dgb, left_on='lowerUserName', right_on='author_email', suffixes=('', '_byuser'),
                      how='left')
    result = pd.merge(result, djira, left_on='lowerUserName', right_on='reporter', suffixes=('', '_Reports'),
                      how='left')

    result.commits.fillna(0.0, inplace=True)
    result.commits_byuser.fillna(0.0, inplace=True)
    result.reports_number.fillna(0.0, inplace=True)
    result['total_commits'] = result['commits'] + result['commits_byuser']
    # normalization
    result['total_commits'] = (result['total_commits'] - result.total_commits.min()) / (
            result.total_commits.max() - result.total_commits.min())
    result['reports_number'] = (result.reports_number - result.reports_number.min()) / (
            result.reports_number.max() - result.reports_number.min())
    ######################
    # generate seniority #
    ######################

    result['seniority'] = (pd.datetime.now().date() - result['createdDate']).dt.days
    # normalize seniority
    result['seniority'] = (result['seniority'] - result.seniority.min()) / (
            result.seniority.max() - result.seniority.min())

    result = cl.drop_columns(result, ['author_email', 'commits', 'author_email_byuser', 'commits_byuser',
                                      'userName', 'createdDate', 'reporter'])

    return result


def main():
    reputation = generate_reputation()
    # reputation = ut.load('../data_out/ReporterReputation.csv')
    ut.save_csv(reputation, '../data_out/ReporterReputation.csv', False)
    # combined = sm.generate()
    combined = ut.load('../data_out/OscarBugSimilarities.csv')
    #ut.save_csv(combined, '../data_out/OscarBugSimilarities.csv', False)
    classified = ut.load('../data_in/OscarDuplicationClassification.csv')
    result = pd.merge(combined, classified, left_on=['bugid_1', 'bugid_2'], right_on=['bugid_1', 'bugid_2'], how='left')
    ut.save_csv(result, '../data_out/OscarBugSimilaritiesClassified.csv', False)

    result = pd.merge(combined, reputation, left_on='reporter', right_on='lowerUserName', how='left')
    cl.drop_columns(result, ['lowerUserName'])
    result = pd.merge(result, classified, left_on=['bugid_1', 'bugid_2'], right_on=['bugid_1', 'bugid_2'], how='left')
    ut.save_csv(result, '../data_out/OscarBugSimilaritiesReporterReputationClassified.csv', False)


if __name__ == '__main__':
    main()
