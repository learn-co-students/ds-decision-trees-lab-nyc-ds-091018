
# An Introduction to Decision Trees

## 1. Game of Thrones example

### 1.1 Generate and pre-process the data

In the following code block, we generate a data set with just one variable, "age", to mimick our "Game of Thrones" tree example. 


```python
import random
import numpy as np
np.random.seed(0)
import pandas as pd

#This code is provided 
random.seed(123)
age_0 = pd.DataFrame([19])
age_1 = pd.DataFrame(sorted(random.sample(range(18, 50), 20)))
age_2 = pd.DataFrame(sorted(random.sample(range(35, 70), 20)))
age_3 = pd.DataFrame([68])

age = age_0.append(age_1, ignore_index= True)
age = age.append(age_2, ignore_index= True)
age = age.append(age_3, ignore_index= True)


label_1 = pd.DataFrame([1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
label_2 = pd.DataFrame([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0])


label = label_1.append(label_2, ignore_index=True)
data = pd.concat([age, label], axis=1)

data.columns = ['age', 'label']
```

This data is created in such a way that the data is not perfectly separable, and also in a way that younger people seem to be more likely to watch the show.

In the cell below, display both the head and the tail of our `data` DataFrame to get a feel for our data. 


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>64</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>68</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>68</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2  Manually create a split

We'll try to manually calculate what the "perfect split" is for this data set, so we'll basically try to recreate the first "split" in the decision tree from our lab. Let's show it again:

![title](G_of_T_tree.png)

Is 41 the best split? Let's find out! You'll create 3 functions in total:
- A function `split()` which splits up the data set in a way that you can easily compute the gini for the two "children" of the parent node, given a split value.
- A function `gini_score()`, which, given a certain split, computes the gini for the left node, the gini for the right node, and the purity gain
- A function `best_split()` which loops over the former two functions to find the best split. 

Let's start by creating the `split()` function. This function takes three arguments, the first one is the column name, which should be a string, the second one is the value that represents the split (in this example, representing a certain age), and as a third argument the name of the data set.  This function is generalized in order to allow us to reuse it for splits on things other than age. 

In the cell below, complete the split function.  This function should split the data from our `data` object just as seen in the Decision Tree diagram above.  

* `data_left` should contain every row where the value for the DataFrame's `col_name` column is less than or equal to the `value` parameter passed into the function. 
* `data_right` should contain every row where the value for the DataFrame's `col_name` column is greater than the `value` parameter passed into the function. 
* The function should return `data_left`, and then `data_right`.


```python
def split(col_name, value, data):
    #split the data in 2 given a column name and the value
    data_left = data.loc[data[col_name] <= value]
    data_right = data.loc[data[col_name] > value]
    return data_left, data_right
```

Let's use our newly created function on the column "age" (for this data the only option), and look at the age 44.

In the cell below, use our newly created `split` function to split our data.  Then, examine `data_left` and `data_right `to confirm that it worked.  


```python
data_left, data_right = split("age", 44, data)
```


```python
data_left
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>26</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>28</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>29</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>37</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>39</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>42</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>44</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>35</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>37</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>39</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>44</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_right
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>46</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>48</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>49</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>45</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>47</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>49</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>53</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>54</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>59</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>64</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>68</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>68</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### GINI Score

Next, we'll complete the `gini_score` function. This function should calculate the gini score. In the real world, just about any Decision Tree algorithm will handle this step for us.  However, we're going to build a function to calculate a gini score for this lab to give ourselves a better understanding of how Decision Trees work.  

Since this function is a bit complicated, comments have been added to help you keep track of what to do at each step.

In a nutshell, to calculate gini, our function will: 
* calculate the size of the data on the left side of the split, the right side of the split, and the combined size of the two. 
* Calculate the chance of seeing each outcome in the left side versus the right side--e.g. if we randomly sample one row from the left side, what is the probability that person watches GoT?
* Sum the squares of each node.
* Calculate the gini for each side (subtract what we calculated in the previous step from 1 for each side)
* Calculate the weighted gini for each size using our raw gini score for each side, the size of the side, and the total number of samples. 
* Calculate the overall gain by summing the weighted gini for both sides. 


```python
def gini_score(data_left, data_right):
    
    # amount of instances flowing in the left vs right node
    size_left= len(data_left)
    size_right = len(data_right)
    n_samples = size_left + size_right
    
    # respective chances of seeing each outcome in the left vs right node 
    p_0L = (data_left['label']==0).sum() / size_left
    p_1L = (data_left['label'] == 1).sum() / size_left
    p_0R = (data_right['label'] == 0).sum() / size_right
    p_1R = (data_right['label'] == 1).sum() / size_right
    
    #take the squares and sum over each node
    score_L = p_0L*p_0L+p_1L*p_1L
    score_R = p_0R*p_0R+p_1R*p_1R
    
    # left node and right node ginis
    gini_L = 1- score_L
    gini_R = 1- score_R
    
    #weighted ginis
    weight_gini_L = gini_L * (size_left / n_samples)
    weight_gini_R = gini_R * (size_right / n_samples)
    
    # The gains (here, we don't compute the root gini again. This value should be mimimized.)
    gain = weight_gini_L + weight_gini_R
    return gini_L, gini_R, gain
```

Now, let's call our `gini_score` function to ensure that it works. Call `gini_score` and pass in `data_left` and `data_right` as parameters.


```python
gini_score(data_left, data_right)
# Expected output (0.4444444444444444, 0.4012345679012346, 0.42592592592592593)
```




    (0.4444444444444444, 0.4012345679012346, 0.42592592592592593)



### Determining the Optimal Split

Now that we have a function to split our data on a given value, and another function to determine how good this split using gini score, we'll write one more function to find the split that produces the best possible gini scores. 

In the cell below, complete the best split function.  As with the previous function, we have included comments to help make coding it less complicated.  

The function should:
* Determine the range of the search space (between the minimum and maximum values that column contains) 
* Iterate through that search space.  For every value:
    * Split the data using our split function
    * Calculate the for each side of the split, as well as the gain
        * If the gain is better (lower) than the current best score, update the `best` values
* When every possible value in search space has been tested, return an `output` dictionary containing the best value to split on, the best gain score, the best splits, and the best gini scores for those groups. 


```python
def best_split(data, col_name):
    # make sure you have the correct range to loop over
    min_val = data[col_name].min()
    max_val = data[col_name].max()
    best_score = 999
    # loop over all the ages 
    for i in range(min_val, max_val):
        data_left, data_right = split(col_name, i, data)
        gini_l, gini_r, gain = gini_score(data_left, data_right)
        # update if gain is lower than any previously observed gain 
        if gain < best_score:
            best_val = i
            best_score = gain
            best_groups = data_left, data_right
            best_ginis= gini_l, gini_r
    output = {}
    # create a dictionary with the best value, the best gain, the best groups and the best ginis
    output['val'] = best_val 
    output['gain'] = best_score
    output['groups'] = best_groups  
    output['ginis'] = best_ginis

    return output
```

Now, let's test that our new function works.  


```python
best_output = best_split(data, "age")
split_age = best_output['val']

split_age # Expected Output: 49
```




    49



### Optional

If you want to see what's going on in each loop of `best_split`, uncomment and run the cell below.


```python
# If you want to see what's going on in each loop...
#min_age=19
#max_age=68
#for i in range(min_age, max_age):
#        data_left, data_right = split("age", i, data)
#        data_l,data_r, gini = gini_score(data_left, data_right)
#        print(i)   
#        print(gini)
```

### 1.3 Use scikit learn and verify answer

We'll use scikit-learn to create a decision tree, and verify that the answer the we found manually above is actually correct.  

Run the cell below to import the `tree` module from sklearn. 


```python
from sklearn.tree import DecisionTreeClassifier
```

Now, create a `DecisionTreeClassifier` object. When creating the tree, set the `criterion` equal to `gini`, and the `max_depth` equal to `1`.


```python
clf_GoT = DecisionTreeClassifier(criterion = "gini", max_depth = 1)
```

Now we can use `clf.fit` with "age" as a first argument and "label" as a second argument. If you only have 1 predictor, you need to reshape your predictor using `.reshape(-1, 1)`. 

Run the cell below to fit our Decision Tree Classifier object to the data. 


```python
GoT_tree = clf_GoT.fit(data['age'].values.reshape(-1, 1), data['label'])
```

### Optional: Visualize the Decision Tree

Some 3rd party libraries make it easy to create a visual representation of our fitted Decision Tree Classifier.  Run the cell below to create the visualization. 

**_NOTE:_**  The code below relies on a library called `graphviz` which is notoriously troublesome to get working on some machines. If the code doesn't work immediately, feel free to try and debug it by googling the error message--if it doesn't work, it's probably only because you need to install a missing dependency. If you can't get the visualization to work, feel free to skip this section!


```python
# import graphviz 
# from sklearn.tree import export_graphviz
# import os
# os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
# GoT_graph = tree.export_graphviz(GoT_tree, out_file=None) 
# graph = graphviz.Source(GoT_graph)
```


```python
# export_graphviz(GoT_tree, out_file="mytree.dot")
# with open("mytree.dot") as f: dot_graph = f.read()
# graphviz.Source(dot_graph)
```

We see that scikit learn generated the same split! Now, let's verify if we computed the correct gini measures. 

## 2. US salaries data set

Now, we'll use our newfound knowledge of Decision Trees to solve a bigger problem on a real world dataset!

### 2.1 Data pre-processing

The salary data set was extracted from the census bureau database and contains salary information. The goal is to use this data set and to try to draw conclusions regarding what drives salaries. More specifically, the target variable is categorical (> 50k; <= 50 k).

Run the cell below to import everything we'll need for this step.  


```python
import pandas as pd
import numpy as np
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns
```

    C:\Users\medio\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

Now, import the data, which is contained in the file `salaries_final.csv`.  Set the `index_col` to `0` when you do this. Display the tail of the new DataFrame to ensure that everything loaded correctly. 


```python
salaries = pd.read_csv("salaries_final.csv", index_col = 0)
```


```python
salaries.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Education</th>
      <th>Occupation</th>
      <th>Relationship</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32556</th>
      <td>27</td>
      <td>Assoc-acdm</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32557</th>
      <td>40</td>
      <td>HS-grad</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32558</th>
      <td>58</td>
      <td>HS-grad</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32559</th>
      <td>22</td>
      <td>HS-grad</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32560</th>
      <td>52</td>
      <td>HS-grad</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>



The dataset "salaries" contains 6 predictors, and one outcome variable, the target salary <= 50k/ >50k.

The 6 predictors are:
- `Age`: continuous.

- `Education`: Categorical. Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, 
Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.

- `Occupation`: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.

- `Relationship`: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.

- `Race`: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.

- `Sex`: Female, Male.

It's important to know that scikit learn needs to get dummies as an input for categorical variables. There are a few ways to do this. Pandas comes prepackaged with the functionality to create dummy variables for any categorical columns (also called **_one-hot encoding_**.  To demonstrate another way to create dummy variables, we've used the `patsy` library. This library borrows syntax from the 'R' language, and denotes each categorical column with a `C`.

Run the cell below to separate our data into a target variable and a one-hot encoded version of the data. 


```python
from patsy import dmatrices
target, data = dmatrices('Target ~ Age + C(Education) + \
                  C(Occupation) + C(Relationship) + C(Race) + C(Sex)',
                  salaries, return_type = "dataframe")
```

In the cell below, split our data into training and testing sets using the appropriate method from sklearn.


```python
data_train, data_test,target_train, target_test = train_test_split(data, target, 
                                                                   test_size = 0.25, random_state=123)
```

### 2.2 Creating Trees

Now that we have prepared our data, we'll create a large Decision Tree to see how it does. 

In the cell below, create a `DecisionTreeClassifier` object, and set the `criterion` parameter to `'gini'`, as well as the `max_depth` parameter to `12`.

Then, run the cell below it to `fit()` our model to the data. 


```python
from sklearn import tree
from sklearn.tree import export_graphviz
sal_tree = tree.DecisionTreeClassifier(criterion = "gini", max_depth = 12)
```


```python
sal_tree = sal_tree.fit(data_train, target_train.iloc[:,1])
```

### Optional

Run the cell below to visualize the fitted Decision Tree Classifier for our salaries dataset.  As before, if you can't seem to get it working, don't worry about it. 


```python
# Optional
# export_graphviz(sal_tree, out_file="mytree.dot", feature_names=data_train.columns , class_names=list(target_train), rounded=True)
# with open("mytree.dot") as f:
#     dot_graph = f.read()
# graphviz.Source(dot_graph)
```

### Creating Smaller Trees

Let's examine if there's a difference with smaller trees.  

In the cell below, create a Decision Tree Classifier as we did before, but this time set the `max_depth` to `3`.  Still set the `criterion` to `gini`. Then, `.fit()` the smaller model to our training data (see the cell above if you are unsure of the syntax).


```python
from sklearn import tree
sal_tree_smaller = tree.DecisionTreeClassifier(criterion = "gini", max_depth = 3)
sal_tree_smaller.fit(data_train, target_train.iloc[:,1])
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')



### Optional

Finally, run the cell below to visualize our smaller Decision Tree with a depth of 3, and examine how it is different.  


```python
# Optional
# export_graphviz(sal_tree_smaller, out_file="mytree.dot", feature_names=data_train.columns ,
#                 class_names=list(target_train), rounded=True)
# with open("mytree.dot") as f:
#     dot_graph = f.read()
# graphviz.Source(dot_graph)
```

Most leaf nodes will point to <= 50 k. How is this possible?
--> imbalance! 5865 make more than 50k <-> 18555 make less (~25-75%)

Note how the left nodes always point to "true" and the right nodes to "false".

### 2.3 Model performance

Now that we have trained models, let's evaluate the performance of each.  

Run the cell below to import the `accuracy_score` helper method.  Then, run the cell below to create some sample predictions on our testing data and generate a `confusion_matrix` and a `classification_report` based on the predictions.  


```python
from sklearn.metrics import accuracy_score
```


```python
pred= sal_tree.predict(data_test)
print(metrics.confusion_matrix(target_test.iloc[:,1], pred))
print(metrics.classification_report(target_test.iloc[:,1], pred))
```

    [[5583  582]
     [ 903 1073]]
                 precision    recall  f1-score   support
    
            0.0       0.86      0.91      0.88      6165
            1.0       0.65      0.54      0.59      1976
    
    avg / total       0.81      0.82      0.81      8141
    
    

Now, run the cell below to generate an accuracy score for our predictions. 


```python
large_tree_accuracy = accuracy_score(target_test.iloc[:,1], pred)

print("Accuracy Score for Large Decision Tree: {:.4}%".format(large_tree_accuracy * 100))
```

    Accuracy Score for Large Decision Tree: 81.76%
    

Now, let's repeat what we did above to generate predictions and reports from our smaller Decision Tree Model.  

In the cell below, create predictions on `data_test` from the smaller decison tree model. The code to provide the confusion matrix and classification report is already provided. 


```python
pred_smaller = sal_tree_smaller.predict(data_test)
print(metrics.confusion_matrix(target_test.iloc[:,1], pred_smaller))
print(metrics.classification_report(target_test.iloc[:,1], pred_smaller))
```

    [[5766  399]
     [1417  559]]
                 precision    recall  f1-score   support
    
            0.0       0.80      0.94      0.86      6165
            1.0       0.58      0.28      0.38      1976
    
    avg / total       0.75      0.78      0.75      8141
    
    

Now, use the `accuracy_score` method to generate an accuracy score for the predictions from our smaller Decision Tree model.  If you are unsure of the syntax, look at the previous example above.  


```python
small_tree_accuracy = accuracy_score(target_test.iloc[:,1], pred_smaller)

print("Accuracy Score for Small Decision Tree: {:.4}%".format(small_tree_accuracy * 100))
```

    Accuracy Score for Small Decision Tree: 77.69%
    

### pre-tuning

One of the best ways to tune Decision Trees to prevent **_overfitting_** is to pre-tune the model by providing constraints on certain aspects of the Decision Tree. Decision Trees are famously prone to overfitting, and tuning the model to be more general can help prevent this.  

The following parameters are the most commonly used for tuning.  In a later lab, you'll learn how to automate the search for the best parameters for each. 

- criterion: either gini for gini impurity, or entropy for information gain.
- max_depth: the maximum depth of a tree.
- min_samples_split: minimum amount of samples required to split an internal node.
- min_samples_leaf: The minimum number of samples required to be at a leaf node.
- class_weight: Weights associated with classes.

We'll end this lab by creating a final Decision Tree Classifier, but with some constraints for parameters mentioned above.  

This model should set the following parameters:
* `criterion='gini'`
* `max_depth=12`
* `min_samples_split=300`
* `min_samples_leaf=150`

Then, fit the model.  Optionally, visualize the tree we've created. 


```python
sal_tree_tuned= tree.DecisionTreeClassifier(criterion = "gini",
                                            max_depth = 12, min_samples_split = 300, min_samples_leaf = 150)
sal_tree_tuned.fit(data_train, target_train.iloc[:,1])
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=12,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=150, min_samples_split=300,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




```python
# export_graphviz(sal_tree_tuned, out_file="mytree.dot", feature_names=data_train.columns , 
#                 class_names=list(target_train), rounded=True)
# with open("mytree.dot") as f:
#     dot_graph = f.read()
# graphviz.Source(dot_graph)
```

Finally, let's create some predictions with this model and check the performance of it.


```python
pred_tuned = sal_tree_tuned.predict(data_test)
print(metrics.confusion_matrix(target_test.iloc[:,1], pred_tuned))
print(metrics.classification_report(target_test.iloc[:,1], pred_tuned))

tuned_accuracy_score = accuracy_score(target_test.iloc[:,1], pred_tuned)

print("Accuracy Score for Tuned Model: {:.4}%".format(tuned_accuracy_score * 100))
```

    [[5731  434]
     [1037  939]]
                 precision    recall  f1-score   support
    
            0.0       0.85      0.93      0.89      6165
            1.0       0.68      0.48      0.56      1976
    
    avg / total       0.81      0.82      0.81      8141
    
    Accuracy Score for Tuned Model: 81.93%
    

For this deep tree: better results with pruning!

# Sources
https://www.svds.com/machine-learning-vs-statistics/ 

https://github.com/xbno/Projects/blob/master/Models_Scratch/Decision%20Trees%20from%20scratch.ipynb

https://archive.ics.uci.edu/ml/machine-learning-databases/adult/

https://www.valentinmihov.com/2015/04/17/adult-income-data-set/

http://scikit-learn.org/stable/modules/tree.html
