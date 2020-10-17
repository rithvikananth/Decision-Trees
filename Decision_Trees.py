import random
import numpy as np
from pprint import pprint
import pandas as pd

def generate_td(k, m):
    final = {}
    dataset = []
    header = []
    for i in range(1, k+1):
        header.append(i)
    header.append("Y")
    dataset.append(header)
    w_denominator = 0
    #finding the denominator of the weights
    for d in range(2, k + 1):
        w_denominator += pow(0.9, d)
    #generating the X,Y values as per the mentioned formulas
    for m in range(1, m+1):
        data = []
        x1 = int(random.randint(0, 1))
        #print(x1)
        data.append(x1)
        previous_value = x1
        features_weight = 0
        for i in range(2, k+1):
            a = int(np.random.binomial(1, 0.75, 1))
            #print(a)
            if a == 1:
                d1 = previous_value
            else:
                d1 = 1 - previous_value
            data.append(d1)
            previous_value = d1
        #print(data)
            features_weight += (pow(0.9, i)/w_denominator)*d1
        if features_weight >= 0.5:
            Y = x1
        else:
            Y = 1 - x1
        data.append(Y)
        dataset.append(data)
    columns = dataset[0]
    final_data = pd.DataFrame(dataset[1:], columns=columns)
    return final_data



def entropy(data):
    target = data.columns[-1]
    y_values = data.iloc[:, -1:]
    prob1 = len(y_values[y_values[target] == 1])/len(y_values)
    prob0 = len(y_values[y_values[target] == 0])/len(y_values)
    if prob0 == 0 or prob1 == 0:
        return 0
    entropy = -prob1*np.log2(prob1) - prob0*np.log2(prob0)
    return entropy

def conditional_entropy(data, c):
    # c is the given column
    entropy_conditional = 0
    y_data = data.iloc[:, -1:]
    target = data.columns[-1]
    Y_values = [0, 1]           # y_data[target].unique()
    X_values = [0, 1]              # data[c].unique()
    for i in X_values:
        e_con_for_x = 0
        len_xcon = len(data[c][data[c] == i])
        #length of the dataset where Xi is either 0 or 1,
        #conditional entropy becomes 0 if there are no values for the x
        if len_xcon == 0:
            continue
        for j in Y_values:
            len_ycon = len(data[c][(data[c] == i) & (data[target] == j)])
            #print(len_ycon)
            if len_ycon == 0:
                continue
            p = len_ycon/(len_xcon)
            e_con_for_x += -p*np.log2(p)
        entropy_conditional += (len_xcon/len(data))*e_con_for_x
    return entropy_conditional

def decision_tree(data, features):
    info_gains = []
    tree = 0
    count0 = 0
    count1 = 0
    target = data.columns[-1]
    y_values = data[target].values.tolist()
    # Termination conditions
    if len(data) == 0:
        return 0
    for i in y_values:
        if i == 0:
            count0 += 1
        else:
            count1 += 1
    if len(features) == 0:
        if count1 >= count0:
            return 1
        else:
            return 0
    if len(np.unique(y_values)) <= 1:
        return y_values[0]
    # finding the maximum information gain column from the data
    for col in data.columns[:-1]:
        info_gains.append(entropy(data) - conditional_entropy(data, col))
    # print(info_gains)
    max_gain = data.columns[info_gains.index(max(info_gains))]
    # print(max_gain)
    if tree == 0:
        tree = {}
        tree[max_gain] = {}
    features = data.columns[:-1]
    features = [i for i in features if i != max_gain]
    # split on the max info gain node where its value is 0 and 1
    subtree = decision_tree(data[data[max_gain] == 0], features)
    tree[max_gain][0] = subtree
    subtree = decision_tree(data[data[max_gain] == 1], features)
    tree[max_gain][1] = subtree
    return tree

# training_data = generate_td(10, 30)
# pprint(training_data)
# features = training_data.columns[:-1]
# tree = decision_tree(training_data, features)
# pprint(tree)


def tree_value(instance_data, tree):
    for node in tree.keys():
        value = instance_data[node]
        tree = tree[node][value]
        t_value = 0
        if type(tree) is dict:
            t_value = tree_value(instance_data, tree)
        else:
            t_value = tree
            break
    return t_value

def error(data, tree):
    error = 0
    td_error_calculation = data.iloc[:, :-1]
    for i in range(len(data)):
        decisiontree_ans = tree_value(td_error_calculation.iloc[i], tree)
        actual_value = data.iloc[i]['Y']
        if decisiontree_ans != actual_value:
            error += 1
    return error / len(data)

def typical_error(k, m, tree):
    te = 0
    for i in range(0,200):
        generate_data = generate_td(k,m)
        te += error(generate_data,tree)
    te = te/200
    return te

def error_estimate():
    es_error = 0
    m = 20
    for mvalue in range(0, 10):
        print("M value", m)
        for i in range(0,10):
            data = generate_td(10, m)      # k = 10
            features = data.columns[:-1]
            tree = decision_tree_gini(data, features)            #tree = decision_tree(data, features)
            es_error += error(data, tree) - typical_error(10, m, tree)
        es_error = es_error/10
        print("Difference of training and true", es_error)
        m += 20

# print("Training error",error(training_data, tree))
# print("typical error:", typical_error(4, 30, tree))
# error_estimate()

def gini_index(data, target):
    gini_value = 1 - (len(data[data[target] == 1]) / len(data)) ** 2 - (
            len(data[data[target] == 0]) / len(data)) ** 2
    return gini_value


def decision_tree_gini(data, features):
    tree = 0
    count0 = 0
    count1 = 0
    target = data.columns[-1]
    y_values = data[target].values.tolist()
    # Termination conditions
    if len(data) == 0:
        return 0
    for i in y_values:
        if i == 0:
            count0 += 1
        else:
            count1 += 1
    if len(features) == 0:
        if count1 >= count0:
            return 1
        else:
            return 0
    if len(np.unique(y_values)) <= 1:
        return y_values[0]
    # finding the maximum gini value of the columns
    gini_values = []

    for col in data.columns[:-1]:
        gini_values.append(gini_index(data, col))

    gini = data.columns[gini_values.index(max(gini_values))]
    #print(max_gain)
    if tree == 0:
        tree = {}
        tree[gini] = {}
    features = data.columns[:-1]
    features = [i for i in features if i != gini]
    # split on the max info gain node where its value is 0 and 1
    subtree = decision_tree(data[data[gini] == 0], features)
    tree[gini][0] = subtree
    subtree = decision_tree(data[data[gini] == 1], features)
    tree[gini][1] = subtree
    return tree


# Part-2 of the question

def pruning_data (m):
    # shrinking it and generating Y values as per professors email
    k = 15         # k = 20
    dataset = []
    header = []
    for i in range(0, k+1):
        header.append(i)
    header.append("Y")
    dataset.append(header)
    #generating the X,Y values as per the mentioned formulas
    for m in range(1, m+1):
        data = []
        x0 = int(random.randint(0, 1))
        data.append(x0)
        previous_value = x0
        for i in range(1, k+1):
            k = i
            if k < 11:                                  # k<15
                a = int(np.random.binomial(1, 0.75, 1))
                if a == 1:
                    d1 = previous_value
                else:
                    d1 = 1 - previous_value
            elif k >= 11:                               # k>=15
                d1 = int(np.random.binomial(1, 0.5, 1))
            data.append(d1)
            previous_value = d1
        count0 = 0
        count1 = 0
        if x0 == 0:
            for i in range(1, 5):          #range(1, 8)
                if data[i] == 0:
                    count0 += 1
                else:
                    count1 += 1
            if count0 > count1:
                y = 0
            else:
                y = 1
        else:
            for i in range(6, 11):          #range(8,15)
                if data[i] == 0:
                    count0 += 1
                else:
                    count1 += 1
            if count0 > count1:
                y = 0
            else:
                y = 1
        data.append(y)
        dataset.append(data)
    columns = dataset[0]
    final_data = pd.DataFrame(dataset[1:], columns=columns)
    return final_data

# training_data = pruning_data(30)
# pprint(training_data)
# features = training_data.columns[:-1]
# tree = decision_tree(training_data, features)
# pprint(tree)
# test_data = pruning_data(30)
# print(error(test_data, tree))
# print("typical error:", typical_error(4, 30, tree))
# error_estimate()

def typical_error_pd(m, tree):
    te = 0
    for i in range(0,200):
        generate_data = pruning_data(m)
        te += error(generate_data, tree)
    te = te/200
    return te


def error_estimate_pd():
    es_error = 0
    m = 200
    for mvalue in range(0, 10):
        print("M value", m)
        for i in range(0,10):
            data = pruning_data(m)
            features = data.columns[:-1]
            tree = decision_tree(data, features)
            es_error += error(data, tree) - typical_error_pd(m, tree)
        es_error = es_error/10
        print("Difference of training and true", es_error)
        m += 200

error_estimate_pd()


def tree_variables(data, tree, a = []):
    for node in tree.keys():
        a.append(node)
        value0 = tree[node][0]
        #print("val0", value0)
        if type(value0) is dict:
            tree_variables(data, value0)
        value1 = tree[node][1]
        #print("val1", value1)
        if type(value1) is dict:
            tree_variables(data, value1)
        else:
            break
    return a

def irrilavent_variables():
    m = 1000
    iv_count = 0
    for mvalue in range(0, 10):
        data = pruning_data(m)
        # print(data)
        features = data.columns[:-1]
        tree = decision_tree(data, features)
        tree_variable = tree_variables(data, tree)
        # print(tree_variable)
        b = []
        for variables in tree_variable:
            if variables > 10:
                b.append(variables)
        iv_count += len(b)
        # print("M:", m, "irrelevant variables:", len(b))
        # print("prob of irrelevant variables:", (len(b)) / (len(tree_variable)))
        m += 1000
    return print("Average irrelevant variables in tree:", iv_count/10)

# irrilavent_variables()
# pprint(testing_dataset)
# pprint(len(testing_dataset.columns))

def decision_tree_depth(data, features, depth, d):
    info_gains = []
    tree = 0
    count0 = 0
    count1 = 0
    target = data.columns[-1]
    y_values = data[target].values.tolist()
    # Termination conditions
    if len(data) == 0:
        return 0
    for i in y_values:
        if i == 0:
            count0 += 1
        else:
            count1 += 1
    if len(features) == 0:
        if count1 >= count0:
            return 1
        else:
            return 0
    if len(np.unique(y_values)) <= 1:
        return y_values[0]

    # finding the maximum information gain column from the data
    for col in data.columns[:-1]:
        info_gains.append(entropy(data) - conditional_entropy(data, col))
    # print(info_gains)
    max_gain = data.columns[info_gains.index(max(info_gains))]
    # print(max_gain)
    if tree == 0:
        tree = {}
        tree[max_gain] = {}
    features = data.columns[:-1]
    features = [i for i in features if i != max_gain]
    # split on the max info gain node where its value is 0 and 1
    if d < depth:
        d += 1
        subtree = decision_tree_depth(data[data[max_gain] == 0], features, depth, d)
        tree[max_gain][0] = subtree
        subtree = decision_tree_depth(data[data[max_gain] == 1], features, depth, d)
        tree[max_gain][1] = subtree
        # print(tree)
    else:
        if count1 >= count0:
            tree[max_gain][0] = 1
            tree[max_gain][1] = 1
        else:
            tree[max_gain][0] = 0
            tree[max_gain][1] = 0
    return tree


def pruning_depth_error():
    data = pruning_data(10000)
    training_data = data.iloc[:8000, :]
    testing_data = data.iloc[8000:, :]
    depth = 1
    features = training_data.columns[:-1]
    for i in range(0,10):
        tree = decision_tree_depth(training_data, features, depth, 0)
        print("Depth",depth, "Error of training data", error(training_data, tree))
        print("Testing error:", error(testing_data, tree))
        depth += 1

# pruning_depth_error()


def decision_tree_sample_size(data, features, sample_size):
    info_gains = []
    tree = 0
    count0 = 0
    count1 = 0
    target = data.columns[-1]
    y_values = data[target].values.tolist()
    # Termination conditions
    if len(data) == 0:
        return 0
    for i in y_values:
        if i == 0:
            count0 += 1
        else:
            count1 += 1
    if len(features) == 0:
        if count1 >= count0:
            return 1
        else:
            return 0
    if len(np.unique(y_values)) <= 1:
        return y_values[0]

    # finding the maximum information gain column from the data
    for col in data.columns[:-1]:
        info_gains.append(entropy(data) - conditional_entropy(data, col))
    # print(info_gains)
    max_gain = data.columns[info_gains.index(max(info_gains))]
    # print(max_gain)
    if tree == 0:
        tree = {}
        tree[max_gain] = {}
    features = data.columns[:-1]
    features = [i for i in features if i != max_gain]
    size = len(data)
    # split on the max info gain node where its value is 0 and 1
    if size >= sample_size:
        subtree = decision_tree_sample_size(data[data[max_gain] == 0], features, sample_size)
        tree[max_gain][0] = subtree
        subtree = decision_tree_sample_size(data[data[max_gain] == 1], features, sample_size)
        tree[max_gain][1] = subtree
        # print(tree)
    else:
        if count1 >= count0:
            tree[max_gain][0] = 1
            tree[max_gain][1] = 1
        else:
            tree[max_gain][0] = 0
            tree[max_gain][1] = 0
    return tree

# training_dataset = pruning_data(8000)
# features = training_dataset.columns[:-1]
# tree = decision_tree_sample_size(training_dataset, features, 2000)
# pprint(tree)

def pruning_samplesize_error():
    data = pruning_data(10000)
    training_data = data.iloc[:8000, :]
    testing_data = data.iloc[8000:, :]
    sample_size = 200
    features = training_data.columns[:-1]
    for i in range(0,15):
        tree = decision_tree_sample_size(training_data, features, sample_size)
        print("Sample Size",sample_size, "Error of training data", error(training_data, tree))
        print("Testing error:", error(testing_data, tree))
        sample_size -= 10

# pruning_samplesize_error()


def irrilavent_variables_depth(depth):
    m = 1000
    iv_count = 0
    print("Depth", depth)
    for mvalue in range(0, 10):
        data = pruning_data(m)
        features = data.columns[:-1]
        tree = decision_tree_depth(data, features, depth, 0)
        tree_variable = tree_variables(data, tree)
        b = []
        for variables in tree_variable:
            if variables > 10:
                b.append(variables)
        iv_count += len(b)
        print("M value:", m)
        print("Irrelevant variables", len(b))
        print("prob of irrelevant variables:", len(b)/ len(tree_variable))
        m += 1000

    return print("Average irrelevant variables in tree:", iv_count/10)

# irrilavent_variables_depth(8)   #depth = 8 from 2.3a

def irrilavent_variables_samplesize(sample_size):
    m = 1000
    print(sample_size)
    for i in range(0, 10):
        data = pruning_data(m)
        features = data.columns[:-1]
        tree = decision_tree_sample_size(data, features, sample_size)
        tree_variable = tree_variables(data, tree)
        # print(tree_variable)
        b = []
        for variables in tree_variable:
            if variables > 10:
                b.append(variables)
        print("M value:", m)
        print("prob of irrelevant variables:", len(b) / len(tree_variable))
        m += 1000


# sample_size = 8000
# irrilavent_variables_samplesize(100)













