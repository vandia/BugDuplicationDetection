import io_utilities as ut
import cleaning_utilities as cl
import pandas as pd
import requests
import json
import csv


def generate_reputation():

    generate_sourceforge_user()

    # Creation of reporter social parameters

    due = ut.load('../data_in/OscarUserExperience.csv', date_cols=['createdDate', 'updatedDate'])
    dgit = ut.load('../data_in/OscarGitLog.csv', date_cols=['date'])
    djira = ut.load('../data_in/OscarBugDetails.csv')
    djira['reporter'] = djira.apply(
        lambda row: row['reporter'] if pd.isnull(row['sourceforge_reporter']) else row['sourceforge_reporter'], axis=1)
    ut.save_csv(djira, '../data_in/OscarBugDetails.csv', False)
    dsforge = ut.load('../data_in/OscarSourceForgeUsers.csv', date_cols=['createdDate'])

    due = cl.drop_columns(due, ['directoryId', 'updatedDate', 'displayName', 'lowerDisplayName',
                                'emailAddress', 'credential', 'localServiceDeskUser'])

    dsforge['id'] = dsforge.index + due.id.max()
    #dsforge['lowerEmailAddress']=pd.Series()

    due = due.append(dsforge)





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

def generate_sourceforge_user():
    djira = ut.load('../data_in/OscarBugDetails.csv')
    sf_users=djira.sourceforge_reporter.unique()
    baseurl = "https://sourceforge.net/rest/u/"
    with open('../data_in/OscarSourceForgeUsers.csv', 'w') as csvfile:
        fieldnames = ['lowerUserName', 'createdDate','active']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in sf_users:
            print(str(baseurl) + str(i).replace(" ","%20"))
            r = requests.get(str(baseurl) + str(i).replace(" ","%20"))
            if (r.ok):
                data = json.loads(r.content)
                active = 1 if data['status'] == 'active' else 0
                writer.writerow({'lowerUserName':str(i),'createdDate':data['creation_date'],'active':active})


def main():
    result=generate_reputation()
    ut.save_csv(result, '../data_out/ReporterReputation.csv', False)

if __name__ == '__main__':
    main()