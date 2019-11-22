#!/usr/bin/env python
# coding: utf-8

# # Simulations and analysis
#
# In this notebook, we assess the performance of FL and FL with DP relative to baseline models.

# In[1]:


import cProfile
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import io
from pstats import SortKey
import pstats

from IPython.display import display

from mozfldp.model import SGDModel
from mozfldp.simulation_runner import FLSimulationRunner, SGDSimulationRunner


def load_dataset_from_file(csv_file):
    """Returns the dataset as a pandas DataFrame. """
    return pd.read_csv(csv_file)


def get_dataset_characteristics(df, label_col="label", user_id_col="user_id"):
    feature_cols = df.drop(columns=[label_col, user_id_col]).columns
    class_labels = df[label_col].unique()
    return {
        "n": len(df),
        "num_features": len(feature_cols),
        "num_classes": len(class_labels),
        "num_users": df[user_id_col].nunique(),
        "feature_cols": feature_cols,
        "class_labels": class_labels,
    }


def summarize_dataset(df, df_info):
    display(df.head())
    print("\nNum training examples: {:,}".format(df_info["n"]))
    print("Num features: {:,}".format(df_info["num_features"]))
    print("Num classes: {:,}".format(df_info["num_classes"]))
    print("Num users: {:,}".format(df_info["num_users"]))

    print("\nLabels:")
    (
        dataset_blob.groupby("label")
        .size()
        .plot.barh(legend=False, title="Num examples per label")
    )
    #plt.show()

    print("Users:")
    (
        dataset_blob.groupby(["user_id", "label"])
        .size()
        .reset_index(name="n_examples")
        .pivot("user_id", "label", "n_examples")
        .plot.bar(
            stacked=True,
            title="Distribution of training examples per user",
            figsize=(20, 8),
        )
    )
    #plt.show()

    print("Features:")
    dataset_blob.hist(
        column=df_info["feature_cols"], bins=50, figsize=(20, 10), sharex=True
    )
   #plt.show()


# First, initialize the various components to be used in running simulations.

# In[3]:


BLOB_DATASET_PATH = "../datasets/blob_S20000_L3_F4_U100.csv"
TEST_DATA_PROP = 0.25


# ## Data
#
# We begin with the "blob" dataset, a randomly generated dataset containing all numerical features grouped into meaningful labelled clusters. A baseline predictive model should perform very well on this data. Indiviual training examples were allocated across users unifomrly at random.

# In[4]:


dataset_blob = load_dataset_from_file(BLOB_DATASET_PATH)
dataset_info = get_dataset_characteristics(dataset_blob)


# In[5]:


summarize_dataset(dataset_blob, dataset_info)


# ### Train/test split
#
# Split the dataset by sampling users.

# In[6]:


users_test = np.random.choice(
    dataset_blob["user_id"].unique(),
    size=int(dataset_info["num_users"] * TEST_DATA_PROP),
    replace=False,
)

dataset_test = dataset_blob[dataset_blob["user_id"].isin(users_test)]
dataset_train = dataset_blob[~dataset_blob["user_id"].isin(users_test)]


# In[7]:


print("Num training examples: {:,}".format(len(dataset_train)))
print("Num testing examples: {:,}".format(len(dataset_test)))


# ### Standardize
#
# Center and scale the features to unit variance.

# In[8]:


scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(dataset_train[dataset_info["feature_cols"]])


# In[9]:


for df in [dataset_train, dataset_test]:
    df.loc[:, dataset_info["feature_cols"]] = scaler.transform(
        df.loc[:, dataset_info["feature_cols"]]
    )


# In[10]:


print("Feature means: {}".format(scaler.mean_))
print("Feature standard devs: {}".format(scaler.scale_))


# ## Model
#
# Start with a linear SVM model.

# In[11]:


sgd_model = SGDModel(
    loss="hinge",
    # shuffling shouldn't matter using our minibatch approach, but just in case
    shuffle=False,
    # default learning rate decays with the number of iterations
    learning_rate="optimal",
)


# In[12]:


sgd_model


# ### Initial weights
#
# Select initial model weights uniformly at random from the square with side [-1, 1].

# In[13]:


def select_initial_weights(dataset_info, range_max=1):
    nrows = dataset_info["num_classes"]
    # Special case for 2 classes: single weight vector.
    if nrows == 2:
        nrows = 1
    init_weights = np.random.random_sample((nrows, dataset_info["num_features"] + 1))
    return init_weights[:, :-1], init_weights[:, -1]


# In[14]:


init_coef, init_intercept = select_initial_weights(dataset_info)


# In[15]:


print("Initial coefs")
print(init_coef)
print("\nInitial intercept")
print(init_intercept)


# ## Parameters
#
# Through the simulations we compare Federated Learning to the standard non-federated approach to training the model above.
#
# Model training is performed by iterating through a number of communication rounds. In the federated context, each communication round retrieves model updates across all users and averages them centrally. The non-federated approach doesn't involve a concept of communication rounds beyond training epochs (dataset passes). For the purposes of comparison, a non-federated communication round is considered to be a fixed number of epochs.
#
# Comparions are made across combinations of the following parameters:
#
# - `num_epochs`: Number of passes over the training data (training epochs) in each communication round.
#     * FL: the number of passes each client makes over its training data prior to central averaging
#     * non-FL: the number of passes over the dataset considered as a "round"
#
#
# - `batch_size`: Target number of training examples included per weight update (gradient descent step), aka "minibatch". Data is allocated to batches uniformly at random. Actual batch sizes may be smaller, eg. if the dataset doesn't divide evenly into batches of this size. Standard SGD uses a `batch_size` of 1, and full-batch GD uses $\infty$.
#     * FL: batching is done separately on each client's dataset.
#     * non-FL: batching is applied across the entire dataset.
#
#
# - `client_fraction`: Proportion of clients whose data is included in each communication round.
#     * non-FL: as the data is not considered split by client, the fraction is essentially 1. However, we could consider pooling all the data from a fraction of clients for experimental purposes.
#
#
# - `sensitivity`: (FLDP) the maximal size of a single weight update (GD step), ie. for a single client batch, in terms of vector norm.
#
#
# - `noise_scale`: (FLDP) parameter controlling the tradeoff between the noise applied in each communication round and the allowable number of training rounds falling within privacy budget.
#
#
# - `user_weight_cap`: (FLDP) limit on the influence of a single user's weight update in the federated average. A higher limit requires more noise to be applied.

# In[16]:


model_params = {"num_epochs": 1, "batch_size": 10, "client_fraction": 0.1}


# In[17]:


# Test this out with a reduced dataset temporarily.

users_reduced = np.random.choice(
    dataset_train["user_id"].unique(), size=20, replace=False
)
dataset_reduced = dataset_train[dataset_train["user_id"].isin(users_reduced)]


# In[18]:


sim_sgd = SGDSimulationRunner(
    num_epochs=model_params["num_epochs"],
    batch_size=model_params["batch_size"],
    model=sgd_model,
    training_data=dataset_reduced,
    # dataset_train,
    coef_init=init_coef,
    intercept_init=init_intercept,
)


# In[19]:


sim_fl = FLSimulationRunner(
    num_epochs=model_params["num_epochs"],
    batch_size=model_params["batch_size"],
    client_fraction=model_params["client_fraction"],
    model=sgd_model,
    training_data=dataset_reduced,
    # dataset_train,
    coef_init=init_coef,
    intercept_init=init_intercept,
)


# In[20]:


model_eval = sgd_model.get_clone()
model_eval.set_training_classes(dataset_train["label"])


def compute_accuracy(coef, intercept):
    model_eval.set_weights(coef, intercept)
    return model_eval.classifier.score(
        dataset_test[dataset_info["feature_cols"]], dataset_test["label"]
    )


# In[21]:


NUM_ROUNDS = 20


acc_sgd = []
acc_fl = []


def train():
    for i in range(NUM_ROUNDS):
        coef, intercept = sim_sgd.run_simulation_round()
        acc_sgd.append(compute_accuracy(coef, intercept))
        coef, intercept = sim_fl.run_simulation_round()
        acc_fl.append(compute_accuracy(coef, intercept))

pr = cProfile.Profile() 
pr.enable()
train()
pr.disable()

s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)

ps.print_stats()
print(s.getvalue())

pr.dump_stats('dump.cprofile')
pr.print_stats()
