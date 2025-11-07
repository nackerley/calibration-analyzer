"""Summarize sensor types in IMS."""
import pandas as pd

df = pd.read_csv('IMS.csv')
df['Model'] = df.Model.str.replace(' Feedback', '')
df['Model'] = df.Model.str.replace(' Passive', '')
df[['Name', 'Code']] = df.pop('Assigned Locations').str.split(n=1, expand=True)
df['Code'] = df.Code.str.replace('(', '').str.replace(')', '')
df = df[df['Physical Location Type'] == 'Station - Site']
df.drop(columns=['Physical Location Type', 'PO/Contract Number',
                 'FRD/Release Number', 'No. of Units'], inplace=True)

arrays = df.groupby(['Code']).agg(Count=('Code', 'count'))
arrays = arrays[arrays['Count'] > 3].sort_values('Count', ascending=False)

summary = (df[df.Code.isin(arrays.index)]
           .groupby(['Code', 'Name', 'Type'])
           .agg(
               Models=('Model', lambda series: ','.join(series.unique())),
               Count=('Model', 'count'))
           .reset_index().set_index('Code', drop=True))
summary.sort_values(['Type', 'Count'], ascending=False, inplace=True)

summary = summary[summary['Count'] > 3]

print(summary.to_string())
