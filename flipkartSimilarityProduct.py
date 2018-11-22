
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import re


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[4]:


from sklearn.metrics import pairwise_distances


# In[36]:


from sklearn.preprocessing import Normalizer


# ## Load the data

# In[46]:


df=pd.read_csv("flipkart_com-ecommerce_sample.csv")


# ## Pre Processing

# In[47]:


df.columns


# In[48]:


df=df.drop_duplicates(subset='product_name')


# In[50]:


original_df=df.copy()


# In[51]:


dropCols=['uniq_id','crawl_timestamp','product_url','image','is_FK_Advantage_product','pid']


# In[52]:


df=df.drop(dropCols,axis=1)


# In[53]:


df.isna().sum()


# In[12]:


df.product_rating=df.product_rating.replace("No rating available",0)
df.overall_rating=df.overall_rating.replace("No rating available",0)


# In[13]:


df.product_rating=df.product_rating.astype("float")
df.overall_rating=df.overall_rating.astype("float")


# In[14]:


def removeUnwantedElements(strValue):
#     print(strValue)
    strValue=re.sub("[^a-zA-Z]"," ",strValue)
    strValue=strValue.split()
    newStr="";
    for eachStr in strValue:
        if(len(eachStr)>2):
            newStr=newStr+" "+eachStr
    return newStr


# In[15]:


for col in df.columns:
    if(df[col].dtypes=='object'):
        df[col]=df[col].fillna("")
        df[col]=df[col].apply(lambda rowVal: removeUnwantedElements(rowVal))
    else:
        df[col]=df[col].fillna(0).astype("float")


# ## Converting brand to dummies

# In[16]:


df=pd.concat([df,pd.get_dummies(df.brand)],axis=1)


# In[17]:


df.drop(['brand'],axis=1,inplace=True)


# In[18]:


original_df.columns


# ## Doing TF IDF Vectorizer on each Column

# In[19]:


naCount=df.isna().sum()
naCount[naCount>0]


# In[20]:


df.columns


# In[21]:


vectorColumns=['product_name','product_category_tree','description','product_specifications']


# In[22]:


def vectorizeAndAppendDataFrame(columnName,dataFrame,min_df):
    vectorizer=TfidfVectorizer(analyzer="word",min_df=min_df,stop_words="english")
    x=vectorizer.fit_transform(df[columnName])
    x=x.toarray()
    x=pd.DataFrame(x)
    vocab=vectorizer.get_feature_names()
    x.columns=[columnName+"_"+str(col) for col in x.columns]
    dataFrame=dataFrame.drop([columnName],axis=1)
    dataFrame=pd.concat([dataFrame,x],axis=1)
    return dataFrame,vocab


# In[31]:


df=df.reset_index()
df=df.drop(['index'],axis=1)


# In[32]:


for col in vectorColumns:
    df,vocab=vectorizeAndAppendDataFrame(col,df,0.01)
    print(col,"has vocabulary of ",len(vocab))


# In[33]:


naCount=df.isna().sum()
naCount[naCount>0]


# In[34]:


len(df.columns)


# ## Normalizing Price

# In[37]:


df[['retail_price','discounted_price']]=Normalizer().fit_transform(df[['retail_price','discounted_price']])


# ## Doing pairwise Similarity

# In[40]:


simResult=pairwise_distances(df,metric="cosine")


# In[41]:


similarityDf=pd.DataFrame(simResult)


# In[42]:


similarityDf.shape


# In[54]:


similarityDf.columns=original_df.pid


# In[55]:


similarityDf.index=original_df.pid


# In[57]:


similarityDf.head()


# In[98]:


similarityDf['SRTEH2FF9KEDEFGF'].nsmallest(6).index


# ## Simulating Simple Reco System

# In[65]:


from ipywidgets import widgets


# In[102]:


def getnLargestSimilarItems(product_id):
    return list(similarityDf[str(product_id)].nsmallest(6).index),list(similarityDf[product_id].nsmallest(6).values)

def getProductInfo(product_id):
    itemRow=original_df[original_df.pid==str(product_id)]
    pName=itemRow.product_name
    pDesc=itemRow.description
    pProductSpec=itemRow.product_specifications
    pCat=itemRow.product_category_tree
    return pName,pDesc,pProductSpec,pCat

def printProductInfo(product_id):
    name,desc,spec,cat=getProductInfo(product_id)
    print("Name:",name)
    print("Desc:",desc)
    print("Spec:",spec)
    print("cat:",cat)
    
def getProductIdAndRecommend(sender):
    product_id=sender.value;
    itemRow=original_df[original_df.pid==str(product_id)]
    if(str(product_id) in similarityDf.columns):
        similarProductList,similarityScore=getnLargestSimilarItems(product_id)
        print("Current Products Below::")
        printProductInfo(product_id)
        print("\nPrint Similar Products\n");
        for i in range(len(similarProductList)):
            if(similarProductList[i]!=product_id):
                printProductInfo(similarProductList[i])
                print("Similarity Score of this product is ",(1-similarityScore[i]))
                print("\n\n")


# In[103]:


text=widgets.Text()
display(text)
def handle_submit(asd):
    print(asd.value)
text.on_submit(getProductIdAndRecommend)

