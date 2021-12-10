#!/usr/bin/env python
# coding: utf-8

# # Apriori

# ## Importing the libraries

# In[1]:


get_ipython().system('pip install apyori ')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Data Preprocessing

# In[3]:


dataset = pd.read_csv('Market_Cart_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
for i in range(0,10):
  print(transactions[i])


# ## Training the Apriori model on the dataset

# In[4]:


from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)


# ## Visualising the results

# ### Displaying the first results coming directly from the output of the apriori function
# 

# In[5]:


results = list(rules)
results


# ### Putting the results well organised into a Pandas DataFrame

# In[6]:


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


# ### Displaying the results

# In[7]:


resultsinDataFrame.nlargest(n = 10, columns = 'Lift')


# #Prediction system

# #Building Model

# In[8]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
dataset=[['Bread','Milk','Chips'],
        ['Bread','Choclate','Eggs'],
        ['Milk','Choclate','Chips','Cola'],
        ['Bread','Milk','Choclate','Chips'],
        ['Bread','Milk','Cola']]
te=TransactionEncoder()
te=te.fit(dataset)
te_ary=te.transform(dataset)
df=pd.DataFrame(te_ary,columns=te.columns_)
market=set(te.columns_)
Freq_itenset=apriori(df,min_support=0.5,use_colnames=True)
from mlxtend.frequent_patterns import association_rules
rules=association_rules(Freq_itenset,metric='support',min_threshold=0.5)
rules[rules['support']==rules['support'].max()]


# ##Prediction program

# In[ ]:


print('------------------------------Cart Addition App------------------------------\nPRESS EXIT TO LEAVE')
print('Items which can be bought : %s\nEnter first item :'%(market))
i,item=1,''
cart=set()
sugg=set()
while i>0:
    item=input('Enter an item :')
    item=item[0].upper()+item[1:].lower()
    if item in cart:
        print('already in cart')
    elif item in market:
        item=set([item])
        cart|=item
        market-=item
        g=rules[rules['antecedents'].apply(lambda x:set(item).issubset(set(x)))]
        le=g['consequents'].index
        a=set()
        for o in le:
            t=set(list(g['consequents'][o])); a|=t
        a-=cart
        a-=item
        sugg=a
        print('*********************************************************************')
        print('Your cart consists of', cart)
        print('*********************************************************************')
        print('suggestions : %s'%(sugg))
        print('Remaining items in the market: %s'%(market))
        sugg=set()
        if len(market)==0:
            print('All items are taken')
            print('*********************************************************************')
            print(' final cart is : %s'%(cart))
            print('*********************************************************************')
            break
    elif item=='Exit':
        print('*********************************************************************')
        print('final cart is : %s'%(cart))
        print('*********************************************************************')
        break
    else:
        print('item is not available')
    


# In[ ]:




