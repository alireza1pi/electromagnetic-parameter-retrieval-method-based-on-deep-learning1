import pandas as pd
colnames=['frequency', 'SP_real', 'SP_imag', 'Z','W']
df1=pd.read_csv('new1.csv', names=colnames,header=None )
df2=pd.read_csv('new2.csv', names=colnames,header=None )
frames=[df1,df2]
result1 = pd.concat(frames)
result1 = result1[result1['frequency'].notna()]
result=result1.reset_index()

ep=[]
tan=[]
for i in range(len(result)):
    a=result['frequency'][i]
    if a[0]=='P':
        x=a.find('eps=')
        x1=a[x:].find(';')
        x2=a[x:]
        x3=x2[4:x1]
        y=a.find('tandel=')
        y1=a[y:].find(';')
        y2=a[y:]
        y3=y2[7:y1]
        
        eps=x3
        tandel=y3
        ep=ep+[eps]
        tan=tan+[tandel]
        
tan1=[]
for i in range(len(tan)):
    if tan[i]!= '':
        tan1=tan1+[float(tan[i])]
tan1=tan1+[tan1[len(tan1)-1]]
eps1=[]
for i in range(len(ep)):
    if ep[i]!= '':
        eps1=eps1+[float(ep[i])]
    
eps1=eps1+[eps1[len(eps1)-1]]

data1=result.dropna()
data2=data1.drop(['Z','W','index'],axis=1)
data3=data2.reset_index()
data3['frequency']=pd.to_numeric(data3['frequency'])
data3['SP_real']=pd.to_numeric(data3['SP_real'])
data3['SP_imag']=pd.to_numeric(data3['SP_imag'])
ra1=(max(data3['frequency'])-min(data3['frequency']))/((data3['frequency'][1])-(data3['frequency'][0]))
ra=int(ra1)
j=0
h=0
aa=[]
bb=[]
while j<len(tan1):
    
    for i in range(0,ra+1):
        ff=j*ra+i
        aa=aa+[tan1[j]]
        bb=bb+[eps1[j]]
   
    j=j+1
    
data3['tan_del']=aa[:len(data3)]
data3['epsilon']=bb[:len(data3)]
data=data3
data.to_csv (r'ff.csv',index=False)
