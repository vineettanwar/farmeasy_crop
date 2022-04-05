import pandas as pds
import xlrd

file =("temp.xlsx")
df = pds.read_excel(file)

def temp(State):
    filt=(df['States']==State)
    a=df.loc[filt,"Rain"]
    b=df.loc[filt,"Temp"]
    c=df.loc[filt,"Humidity"]
    return [[r for r in a],[a for a in b],[s for s in c]]