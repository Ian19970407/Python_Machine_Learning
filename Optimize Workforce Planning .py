#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pulp import *
import matplotlib.pyplot as plt
from itertools import chain, repeat


# In[2]:


def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return chain.from_iterable(repeat(tuple(iterable), n))

# Staff needs per Day
n_staff = [40, 50, 48, 60, 52, 35, 30]
# Days of the week
jours = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create circular list of days
n_days = [i for i in range(7)]
n_days_c = list(ncycles(n_days, 3)) 

# Working days
list_in = [[n_days_c[j] for j in range(i , i + 5)] for i in n_days_c]

# Workers off by shift for each day
list_excl = [[n_days_c[j] for j in range(i + 1, i + 3)] for i in n_days_c]


# In[3]:


list_excl


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

demand = pd.DataFrame(n_staff,index=jours,columns=['Staff Demand'])
plt.figure(figsize=(10,6))
sns.barplot(x=demand.index,y=demand['Staff Demand'])


# In[5]:


# Initialize Model
model = LpProblem("Minimize Staffing", LpMinimize)

# Create Variables
start_jours = ['Shift: ' + i for i in jours]
x = LpVariable.dicts('shift_', n_days, lowBound=0, cat='Integer')

# Define Objective
model += lpSum([x[i] for i in n_days])

# Add constraints
for d, l_excl, staff in zip(n_days, list_excl, n_staff):
    model += lpSum([x[i] for i in n_days if i not in l_excl]) >= staff


# In[6]:


# Solve Model
model.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[model.status])

# How many workers per day ?
dct_work = {}
for v in model.variables():
    dct_work[int(v.name[-1])] = int(v.varValue)
    
# Show Detailed Sizing per Day
dict_sch = {}
for day in dct_work.keys():
    dict_sch[day] = [dct_work[day] if i in list_in[day] else 0 for i in n_days]
df_sch = pd.DataFrame(dict_sch).T
df_sch.columns = jours
df_sch.index = start_jours
df_sch


# In[7]:


# The optimized objective function value
print("Total number of Staff = ", pulp.value(model.objective))


# In[8]:


supply = df_sch.sum(axis=0)
supply = pd.DataFrame(supply,columns=['Staff Supply'])
supply


# In[9]:


demand.index = supply.index

df = pd.concat([demand,supply],axis=1)
df['Extra_Ressources'] = df['Staff Supply'] - df['Staff Demand']
df = df.reset_index()
df


# In[10]:


ax = df[['index','Extra_Ressources']].plot(x='index',linestyle='-',marker='o',color='red')
df[['index', 'Staff Demand', 'Staff Supply']].plot(x='index',kind='bar',ax=ax)
plt.show()

