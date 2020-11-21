#SKAI

## Hackathon 2020

This submission only deals with task 1 of the hackathon. 

# We worked on the CNN model. We spent a lot of time on cleaning data, being non-German speakers, we needed more time to understand the text and clean it. 

# First part of the code deals with cleaning the data and getting the correct data for predictions. 

# For our model 

We used a CNN with the following parameters:

    n_filters = 32
    n_kernal_size = 8
    n_dropout = 0.5
    optim = 'adam'
    n_batch_size = 256
    n_epochs = 100

The challenge was to get the services as well as manuals correct. 

1. We loaded dataset and cleaned the data
2. We then converted list of words to list of indexes which will be converted to one hot encoding later.
3.  Prepare numpy array for training, dev and test data

Model:
1. CNN classifier
2. We used categorical_crossentropy loss
3. Number of Epochs: 100






