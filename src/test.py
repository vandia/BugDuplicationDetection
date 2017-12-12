import numpy
import io_utilities as ut
import cleaning_utilities as cl

details = ut.load('../data_in/OscarBugDetails.csv', date_cols=['creation_date'])
details.sort_values('creation_date', ascending=True, kind='mergesort', inplace=True)
details=details[details['classifier']=='DUPLICATE']
details=cl.drop_columns(details, ['project','number','creation_date','reporter','sourceforge_reporter','type','priority','workflowId','status'])
details.rename(columns = {'bugid':'bugid_2'}, inplace = True)
details.insert(0, 'bugid_1', '')
ut.save_csv(details, '../data_in/OscarDuplicationClassification_Original.csv', False)