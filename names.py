import os 
os.chdir('C:\\Users\\VINEET\\Downloads\cropfit-main\static\images\\')
print(os.getcwd())
for f in (os.listdir()):
    f_name,f_ext=os.path.splitext(f)
    f_name=f_name.upper()
    new_name=f'{f_name}{f_ext}'
    os.rename(f,new_name)