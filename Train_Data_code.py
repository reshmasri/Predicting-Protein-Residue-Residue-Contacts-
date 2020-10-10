#!/usr/bin/env python
# coding: utf-8

# In[1]:



def clear_all():
    """Clears all the variables from the workspace """
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
if __name__ == "__main__":
    clear_all()


# In[19]:


""" Required libraries"""
import numpy as np
import pandas as pd
import urllib.request
import os
import gzip 
import shutil
import matplotlib.pyplot as plt
from itertools import product
import numpy
import itertools
from Bio.PDB.PDBParser import PDBParser
import Bio.PDB
from Bio.PDB.Polypeptide import PPBuilder
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from imblearn.under_sampling import RandomUnderSampler


# In[20]:


""" Train Data"""

path_data= 'train_data.txt'
data = pd.read_csv(path_data)
server ="https://files.rcsb.org/download/" 
exstention = '.pdb.gz'
data['URL'] = data.apply(lambda row: server + row.PDB_id +  exstention, axis = 1) 


path="Dataset_train"
if not os.path.exists(path):
    os.makedirs(path)


# # Downloading Data

# In[21]:


for i in range(len(data)):
    fullfilename = os.path.join(path,  data.PDB_id[i] +'.pdb.gz')
    urllib.request.urlretrieve(data.URL[i], fullfilename)
    with gzip.open(fullfilename, 'rb') as f_in:
        fullfilename1 = os.path.join(path, data.PDB_id[i] +'.pdb')
        with open(fullfilename1, 'wb') as f_out:  
            shutil.copyfileobj(f_in, f_out)


# # Function to craete a dataframe with required features by parsing the PDB file

# In[6]:


def parse_for_distance(pdb_code, pdb_filename):
    parser = PDBParser()
    structure = parser.get_structure(pdb_code, pdb_filename)
    residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
    Residue_one = []
    Residue_two = []
    Residue_one_name = []
    Residue_two_name = []
    Residue_distance = []
    Residue_one_window_10 =[]
    Residue_one_window_20 = []
    Residue_one_window_30 = []
    Residue_two_window_10 =[]
    Residue_two_window_20 = []
    Residue_two_window_30 = []
    for each in itertools.combinations(residues,2):
        one  = each[0]["CA"].get_coord()
        Residue_one.append(each[0].get_id()[1])
        Residue_one_name.append(each[0].get_resname()[0])
        two = each[1]["CA"].get_coord()
        Residue_two.append(each[1].get_id()[1])
        Residue_two_name.append(each[1].get_resname()[0])
        Residue_distance.append(numpy.linalg.norm(one-two))
    pdb_file = [pdb_code] * len(Residue_two)
    
    window_10 = residue(residues,5)
   # print(window_10)
    window_20 = residue(residues,10)
    window_30 = residue(residues,15)
    
    for i in Residue_one:
        if(i in window_10.keys()):
            Residue_one_window_10.append(window_10.get(i))
        if(i in window_20.keys()):
            Residue_one_window_20.append(window_20.get(i))
        if(i in window_30.keys()):
            Residue_one_window_30.append(window_30.get(i))
    for i in Residue_two:
        if(i in window_10.keys()):
            Residue_two_window_10.append(window_10.get(i))
        if(i in window_20.keys()):
            Residue_two_window_20.append(window_20.get(i))
        if(i in window_30.keys()):
            Residue_two_window_30.append(window_30.get(i))
    
    df = pd.DataFrame(
       {'pdb_file': pdb_file,
        'Residue_one': Residue_one,
        'Residue_one_name': Residue_one_name,
        'Residue_two': Residue_two,
        'Residue_two_name': Residue_two_name,
        'Residue_distance': Residue_distance,
        'Residue_one_window_10':Residue_one_window_10,
        'Residue_two_window_10':Residue_two_window_10,
        'Residue_one_window_20':Residue_one_window_20,
        'Residue_two_window_20':Residue_two_window_20,
        'Residue_one_window_30':Residue_one_window_30,
        'Residue_two_window_30':Residue_two_window_30
        })        

    return df


# # Function to read around an amino acid in the residues

# In[7]:


def residue(x,l):
#     l = int(n/2)
    encoding_one = []
    for i in range(len(x)):         
        encoding = []
        residue_id = [y.get_id()[1] for y in x]
        right = [y.get_resname()[1] for y in x[i+1:i+(l+1)]]
        if( i >l):
            left = [y.get_resname()[1] for y in x[i-(l):i]]
        else :
            left = [y.get_resname()[1] for y in x[0:i]] 
        encoding = ''.join(list(itertools.chain(left, right)))
        encoding_one.append(encoding)
    encoding = dict(zip(residue_id, encoding_one))
    return encoding


# # One hot Encoding

# In[15]:


def ohe_vector(t):
    r = []
    test = list("_ACDEFGHIKLMNPQRSTVWY") 
""" 20 naturally occuring amino acids and padding """
    for j in t:
        for i in test:
            if(j == i):
                r.append(1)
            else:
                r.append(0)
    r_array = np.asarray(r)
    return r_array


# # Under Sampling

# In[10]:


def undersampleData(train_X, train_y):
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(train_X, train_y)
    return X_resampled, y_resampled


# In[9]:


Listdf_for_one_pdb = []
for j in range(len(data)):   
    pdb_code  = data.PDB_id[j] 
    pdb_filename = os.path.join(path, data.PDB_id[j] +'.pdb')
    df_for_one_pdb = parse_for_distance(pdb_code, pdb_filename)
    Listdf_for_one_pdb.append(df_for_one_pdb)
Data_all_pdb = pd.concat(Listdf_for_one_pdb)
Data_all_pdb = Data_all_pdb.reset_index(drop=True)


# # Pre-Processing

# In[10]:


Data_all_pdb['Residue_one_window_10'] = Data_all_pdb['Residue_one_window_10'].str.pad(10, side ='left', fillchar ='_')
Data_all_pdb['Residue_two_window_10'] = Data_all_pdb['Residue_two_window_10'].str.pad(10, side ='left', fillchar ='_')
Data_all_pdb['Residue_one_window_20'] = Data_all_pdb['Residue_one_window_20'].str.pad(20, side ='left', fillchar ='_')
Data_all_pdb['Residue_two_window_20'] = Data_all_pdb['Residue_two_window_20'].str.pad(20, side ='left', fillchar ='_')
Data_all_pdb['Residue_one_window_30'] = Data_all_pdb['Residue_one_window_30'].str.pad(30, side ='left', fillchar ='_')
Data_all_pdb['Residue_two_window_30'] = Data_all_pdb['Residue_two_window_30'].str.pad(30, side ='left', fillchar ='_')
Data_all_pdb['Window_10'] = Data_all_pdb[['Residue_one_window_10', 'Residue_two_window_10']].apply(lambda x: ''.join(x), axis = 1)
Data_all_pdb['Window_20'] = Data_all_pdb[['Residue_one_window_20', 'Residue_two_window_20']].apply(lambda x: ''.join(x), axis = 1)
Data_all_pdb['Window_30'] = Data_all_pdb[['Residue_one_window_30', 'Residue_two_window_30']].apply(lambda x: ''.join(x), axis = 1)


# In[11]:


Data_all_pdb['Range'] = Data_all_pdb.Residue_distance.apply(lambda x : "short" if x < 10 else ("medium" if 10 <= x < 20 else "high"))
Data_all_pdb['Range_label'] = Data_all_pdb.Residue_distance.apply(lambda x : 0 if x < 10 else (1 if 10 <= x < 20 else 2))
Data_all_pdb['Contact'] = Data_all_pdb.Residue_distance.apply(lambda x : "Contact" if x <= 8 else "No Contact")
Data_all_pdb['Contact_label'] = Data_all_pdb.Residue_distance.apply(lambda x : 1 if x <= 8 else 0)


# In[12]:


Data_all_pdb.drop(["Residue_one_window_10","Residue_one_window_20","Residue_one_window_30","Residue_two_window_10","Residue_two_window_20","Residue_two_window_30","pdb_file","Residue_one_name",],axis=1,inplace=True)


# # Saving the data into CSV after pre-processing

# In[13]:


Data_with_range = Data_all_pdb.to_csv (r'Data_train.csv', index = None, header=True) 


# In[11]:


Data_all_pdb_path = 'Data_train.csv'
Data_all_pdb = pd.read_csv(Data_all_pdb_path)


# # Exploratory Data Analysis

# In[14]:


print("Shape of Data:",Data_all_pdb.shape)
sns.countplot(Data_all_pdb['Contact'],label="Count")
plt.show()
sns.countplot(Data_all_pdb['Range'],label="Count")
plt.show()


# In[15]:


Contact0 = Data_all_pdb[ Data_all_pdb['Contact'] == "No Contact" ]
sns.countplot(Contact0['Range'],label="Count")
print("Count of no-contact labels:",len(Contact0))
print("Range of no-contact labels:")
plt.show()
Contact1 = Data_all_pdb[ Data_all_pdb['Contact'] == "Contact" ]
print("Count of contact labels:",len(Contact1))
print("Range of contact labels:")
sns.countplot(Contact1['Range'],label="Count")
plt.show()


# In[16]:


Range_short = Data_all_pdb[ Data_all_pdb['Range'] == "short" ]
print("contacts in range labels: Short")
sns.countplot(Range_short['Contact'],label="Count")
plt.show()

Range_medium = Data_all_pdb[ Data_all_pdb['Range'] == "medium" ]
print("contacts in range labels: Medium")
sns.countplot(Range_medium['Contact'],label="Count")
plt.show()

Range_high = Data_all_pdb[ Data_all_pdb['Range'] == "high" ]
print("contacts in range labels: High")
sns.countplot(Range_high['Contact'],label="Count")
plt.show()


# # Under-sampling

# In[12]:


X = Data_all_pdb[["Window_10","Window_20","Window_30","Range_label","Range","Contact"]].values
y = Data_all_pdb["Contact_label"].values
X_resampled, y_resampled = undersampleData(X, y)


# In[13]:


X_window_10 = [i[0] for i in X_resampled]
X_window_20 = [i[1] for i in X_resampled]
X_window_30 = [i[2] for i in X_resampled]
X_Range_label = [i[3] for i in X_resampled]
X_Range = [i[4] for i in X_resampled]
Contact = [i[5] for i in X_resampled]

Data = pd.DataFrame(
       {'Window_10': X_window_10,
        'Window_20': X_window_20,
        'Window_30': X_window_30,
        'Range_label': X_Range_label,
        "Range":X_Range,
        'Contact_label': y_resampled,
        'Contact': Contact
        })


# In[19]:


print("Shape of Data:",Data.shape)
sns.countplot(Data['Contact'],label="Count")
Contact0 = Data[ Data['Contact'] == "No Contact" ]
print("Count of no-contact labels:",len(Contact0))
Contact1 = Data[ Data['Contact'] == "Contact" ]
print("Count of contact labels:",len(Contact1))
plt.show()
sns.countplot(Data['Range'],label="Count")
plt.show()


# # One-Hot encoding

# In[14]:


Data['Window_10_ohe'] = Data.Window_10.apply(ohe_vector)
Data['Window_20_ohe'] = Data.Window_20.apply(ohe_vector)
Data['Window_30_ohe'] = Data.Window_30.apply(ohe_vector)


# # Feature Vector for Window size 10

# In[16]:


x_10 = np.row_stack((Data['Window_10_ohe'][0],Data['Window_10_ohe'][1])) 
for i in range(2,len(Data['Window_10_ohe'])):
    x_10 = np.row_stack((x_10,Data['Window_10_ohe'][i]))
    print(x_10)


# In[ ]:


np.savetxt("X_10_train.txt", x_10,fmt="%s")


# # Feature Vector for Window size 20

# In[15]:


x_20 = np.row_stack((Data['Window_20_ohe'][0],Data['Window_20_ohe'][1])) 
for i in range(2,len(Data['Window_20_ohe'])):
    x_20 = np.row_stack((x_20,Data['Window_20_ohe'][i]))
    print(x_20)
    


# In[16]:


np.savetxt("X_20_train.txt", x_20,fmt="%s")


# # Feature Vector for Window size 30

# In[18]:


x_30 = np.row_stack((Data['Window_30_ohe'][0],Data['Window_30_ohe'][1])) 
for i in range(2,len(Data['Window_30_ohe'])):
    x_30 = np.row_stack((x_30,Data['Window_30_ohe'][i]))
    print(x_30)
    


# In[19]:


np.savetxt("X_30_train.txt", x_30,fmt="%s")


# # Feature Range and Target Variable

# In[14]:


ran = Data['Range_label'].values
np.savetxt("range_train.txt", ran,fmt="%s")


# In[13]:


y_test = Data['Contact_label'].values
np.savetxt("y_train.txt", y_test,fmt="%s")


# In[ ]:





# In[ ]:




