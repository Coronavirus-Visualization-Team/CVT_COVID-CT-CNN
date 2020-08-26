#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os, shutil, random

"""
|
|+ master-data
    |
    |+COVID
    |+NonCOVID
|+ split-data
        |
        |+train
            |
            |+COVID
            |+NonCOVID
        |+test
            |
            |+COVID
            |+NonCOVID
        |+validation
            |
            |+COVID
            |+NonCOVID
"""


positives = os.listdir('master-data/COVID')
negatives = os.listdir('master-data/NonCOVID')

train_percentage = 0.7

    
    
def get_split(original, train_percentage):
    
    """Returns train, validation, test splits (in that order) as randomly created lists of file names.
        - len(train) = train_percentage * len(original)
        - len(val) = len(test) = (1 - train_percentage)/2 * len(original)
        - len(original) = len(train) + len(test) + len(val)

    Parameters
    ----------
    original : list
        list of file names
    train_percentage : float
        percentage of original data that will be made training set
        
    """

    random.shuffle(original) 
    n = len(original)
    
    train_index = int(train_percentage*n)
    val_index = int((n - train_index)/2) + train_index
    
  
    train = original[:train_index]
    val = original[train_index:val_index]
    test = original[val_index:]
    
    return train, val, test


def print_status():
    print()
    print("**Current Status**")
    print("master-data/COVID -- " + str(len(os.listdir("master-data/COVID"))) + " images")
    print("master-data/NonCOVID -- " + str(len(os.listdir("master-data/NonCOVID"))) + " images")
    print("split-data/train/COVID -- " + str(len(os.listdir("split-data/train/COVID"))) + " images")
    print("split-data/validation/COVID -- " + str(len(os.listdir("split-data/validation/COVID"))) + " images")
    print("split-data/test/COVID -- " + str(len(os.listdir("split-data/test/COVID"))) + " images")
    print("split-data/train/NonCOVID -- " + str(len(os.listdir("split-data/train/NonCOVID"))) + " images")
    print("split-data/validation/NonCOVID -- " + str(len(os.listdir("split-data/validation/NonCOVID"))) + " images")
    print("split-data/test/NonCOVID -- " + str(len(os.listdir("split-data/test/NonCOVID"))) + " images")
    print()

if __name__ == '__main__':
    print_status()
    
    #clears contents of folders currently so that a new split can be created
    #positives
    [os.remove('split-data/train/COVID/'+ f) for f in os.listdir('split-data/train/COVID')]
    [os.remove('split-data/validation/COVID/'+ f) for f in os.listdir('split-data/validation/COVID')]
    [os.remove('split-data/test/COVID/'+ f) for f in os.listdir('split-data/test/COVID')]
    #negatives
    [os.remove('split-data/train/NonCOVID/'+ f) for f in os.listdir('split-data/train/NonCOVID')]
    [os.remove('split-data/validation/NonCOVID/'+ f) for f in os.listdir('split-data/validation/NonCOVID')]
    [os.remove('split-data/test/NonCOVID/'+ f) for f in os.listdir('split-data/test/NonCOVID')]
    print("**cleared contents of split-data subdirectories**")
    print_status()
    
    #adds the files to respective folders according to split determined by get_split
    pos_train_set, pos_val_set, pos_test_set = get_split(positives, train_percentage)
    neg_train_set, neg_val_set, neg_test_set = get_split(negatives, train_percentage)
    
    #positives
    [shutil.copy('master-data/COVID/' + f, 'split-data/train/COVID/'+ f) for f in pos_train_set]
    [shutil.copy('master-data/COVID/' + f, 'split-data/validation/COVID/'+ f) for f in pos_val_set]
    [shutil.copy('master-data/COVID/' + f, 'split-data/test/COVID/'+ f) for f in pos_test_set]
    #negatives
    [shutil.copy('master-data/NonCOVID/' + f, 'split-data/train/NonCOVID/'+ f) for f in neg_train_set]
    [shutil.copy('master-data/NonCOVID/' + f, 'split-data/validation/NonCOVID/'+ f) for f in neg_val_set]
    [shutil.copy('master-data/NonCOVID/' + f, 'split-data/test/NonCOVID/'+ f) for f in neg_test_set]
    
    print("**added shuffled data to split-data subdirectories**")

    print_status()

    print("**done**")

    


# In[ ]:




