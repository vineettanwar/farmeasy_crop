import pandas as pds
import xlrd

file =("soil.xlsx")
df = pds.read_excel(file)

def soil(State, District):
    for i in df['District']:
        if i == District:
            filt=((df['State']==State) &(df['District']==District))
            a=df.loc[filt,"Address"]
            return [r for r in a]
        else:
            filt=(df['State']==State)
            a=df.loc[filt,"Address"]
            return [r for r in a]
