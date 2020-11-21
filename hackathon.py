# -*- coding: utf-8 -*-

# eda.py
import csv
import emoji
import random
import numpy as np
import pandas as pd
from string import punctuation, whitespace, digits

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model, load_model
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Embedding
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D
from keras.utils import np_utils
from sklearn.metrics import f1_score


url_train_dev = 'train.csv'
url_test = 'test_reduced.csv'
service_catgs = set()
manual_catgs = set()
vocab = set()

class EmailClassifier(object):
    def remove_emoji(self, text):
        return emoji.get_emoji_regexp().sub(u'', text)

    def clean_rec(self, text):
        # Remove emojis
        doc = self.remove_emoji(text)
        # Remove all words starting with @
        doc = " ".join(filter(lambda x: not x.startswith('@'), doc.split()))
        # Remove all words starting with http:
        doc = " ".join(filter(lambda x: not x.startswith('http:'), doc.split()))
        # Remove all words starting with FW:
        doc = " ".join(filter(lambda x: not x.startswith('FW:'), doc.split()))
        # Remove all words starting with MD5:
        doc = " ".join(filter(lambda x: not x.startswith('MD5:'), doc.split()))
        # Remove all words starting with tel:
        doc = " ".join(filter(lambda x: not x.startswith('tel:'), doc.split()))
        # Remove all words starting with fax:
        doc = " ".join(filter(lambda x: not x.startswith('fax:'), doc.split()))
        # Remove all words starting with website:
        doc = " ".join(filter(lambda x: not x.startswith('website:'), doc.split()))
        # Remove all words starting with Website:
        doc = " ".join(filter(lambda x: not x.startswith('Website:'), doc.split()))
        # Remove all words starting with [cid]:
        doc = " ".join(filter(lambda x: not x.startswith('[cid:'), doc.split()))
        # Remove all words starting with *
        doc = " ".join(filter(lambda x: not x.startswith('*'), doc.split()))
        # Remove all words starting with -
        doc = " ".join(filter(lambda x: not x.startswith('-'), doc.split()))
        # Remove all words starting with _
        doc = " ".join(filter(lambda x: not x.startswith('_'), doc.split()))
        # Remove all words starting with TC
        doc = " ".join(filter(lambda x: not x.startswith('TC'), doc.split()))
        # Remove all words starting with RE:
        doc = " ".join(filter(lambda x: not x.startswith('RE:'), doc.split()))
        # Remove all words starting with %
        doc = " ".join(filter(lambda x: not x.startswith('%'), doc.split()))
        # Remove all words starting with >>
        doc = " ".join(filter(lambda x: not x.startswith('>>'), doc.split()))
        # split into char tokens
        tokens = list(doc)
        # remove punctuation from each token.	
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove whitespace from each token
        table = str.maketrans('', '', whitespace)
        tokens = [w.translate(table) for w in tokens]
        # remove digits from each token
        table = str.maketrans('', '', digits)
        tokens = [w.translate(table) for w in tokens]
        # rejoin tokens from list to a string
        tokens = ' '.join(sorted(list(set(tokens))))
        return tokens

    # load data from files
    def fetch_dataset(self, url, is_train=False):
        i = 0
        emails = []
        with open(url, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader)
            for line in reader:
                i += 1
                if is_train:
                    _, id, impact, urgency, incidentype, serviceprocessed, mailsubject, mailtextbody, manualgroups = line
                    service_catgs.add(serviceprocessed)
                    manual_catgs.add(manualgroups)
                else:
                    _, id, mailsubject, mailtextbody = line

                text = self.clean_rec(mailsubject+mailtextbody).split()
                vocab.update(text)

                if is_train is True:
                    emails.append((text, serviceprocessed, manualgroups))
                else:
                    emails.append((text, ))
        # Reshuffle train data so that a proper train/dev split can be obtained
        if is_train:
            random.shuffle(emails)
        return i, emails

    # Convert list of words to list of indexes which will be converted to one hot encoding later
    def convert_word_to_index(self, word_list, vocab_list):
        out_list = []
        err_index = len(vocab_list)
        for l in word_list:
            try:
                if isinstance(l, list):
                    i = []
                    for j in l:
                        i.append(vocab_list.index(j))
                else:
                    i = vocab_list.index(l)
            except:
                i = err_index
            out_list.append(i)
        return out_list

    # Prepare numpy array for training, dev and test data
    def prepare_x(self, input_X, rec_cnt, vocab_size, vocab_list):
        dataX = np.zeros((rec_cnt, vocab_size))
        for i in range(rec_cnt):
            rec = input_X[i]
            out = self.convert_word_to_index(rec, vocab_list)
            for j in range(len(out)):
                dataX[i][j] = out[j]
        print(dataX.shape)
        return dataX

    def label_num_to_str(self, list_y, total_y):
        out = []
        for i in list_y:
            try:
                s = total_y[i]
            except:
                s = ''
            out.append(s)
        return out
        
    def define_model(self, vocab_size, n_classes, n_filters, n_kernal_size, n_dropout, optim):
        model = Sequential()
        model.add(Input(shape=(vocab_size,)))
        model.add(Embedding(vocab_size, 100))
        model.add(Conv1D(filters=n_filters, kernel_size=n_kernal_size, activation='relu'))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dense(250, activation='relu'))
        model.add(Dropout(n_dropout))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optim)
        print(model.summary())
        return model

if __name__ == '__main__':
    ec = EmailClassifier()
    # Fetch data from files
    # train_cnt - largest vocabulary in train_dev data
    # test_cnt - largest vocabulary in test data
    # X_y_train_dev - list of tuples (squeezed text, label) from train_dev data
    # X_y_test - list of tuples (squeezed text, label) from test data
    train_cnt, X_y_train_dev = ec.fetch_dataset(url_train_dev, is_train=True)
    test_cnt, X_y_test = ec.fetch_dataset(url_test)

    total_service_y = sorted(list(service_catgs))
    n_service_classes = len(service_catgs) + 1

    total_manual_y = sorted(list(manual_catgs))
    n_manual_classes = len(manual_catgs) + 1

    train_cnt = int(train_cnt * 0.8)
    vocab_size = len(vocab)
    vocab_list = sorted(list(vocab))

    print(f"Total training+dev data: {train_cnt}")
    print(f"Total test data:         {test_cnt}")
    print(f"Total service catgs:     {n_service_classes}")
    print(f"Total manual groups:     {n_manual_classes}")
    print(f"Vocab size:              {vocab_size}")

    # Split train_dev data into train and dev sets - X, y data
    X_train = []
    y_train_service = []
    y_train_manual = []
    X_dev = []
    y_dev_service = []
    y_dev_manual = []
    cnt = 0
    for x, y_service, y_manual in X_y_train_dev:
        if cnt <= train_cnt-1:
            X_train.append(x)
            y_train_service.append(y_service)
            y_train_manual.append(y_manual)
        else:
            X_dev.append(x)
            y_dev_service.append(y_service)
            y_dev_manual.append(y_manual)
        cnt += 1

    # Separate X, from test data. There is no y in test data.
    X_test = []
    for x, in X_y_test:
        X_test.append(x)

    # Convert X data to numpy arrays
    trainX = ec.prepare_x(X_train, train_cnt, vocab_size, vocab_list)
    devX = ec.prepare_x(X_dev, len(X_y_train_dev)-train_cnt, vocab_size, vocab_list)
    testX = ec.prepare_x(X_test, len(X_test), vocab_size, vocab_list)

    # Categorize y data
    y_train_service = ec.convert_word_to_index(y_train_service, total_service_y)
    trainy_service = np_utils.to_categorical(y_train_service, n_service_classes)

    y_train_manual = ec.convert_word_to_index(y_train_manual, total_manual_y)
    trainy_manual = np_utils.to_categorical(y_train_manual, n_manual_classes)

    y_dev_service = ec.convert_word_to_index(y_dev_service, total_service_y)
    devy_service = np_utils.to_categorical(y_dev_service, n_service_classes)

    y_dev_manual = ec.convert_word_to_index(y_dev_manual, total_manual_y)
    devy_manual = np_utils.to_categorical(y_dev_manual, n_manual_classes)

    n_filters = 32
    n_kernal_size = 8
    n_dropout = 0.5
    optim = 'adam'
    n_batch_size = 256
    n_epochs = 100

    # define service model
    model_type = 'service'
    model = ec.define_model(vocab_size, n_service_classes, n_filters, n_kernal_size, n_dropout, optim)
    history = model.fit(trainX, trainy_service, verbose=1, batch_size=n_batch_size, epochs=n_epochs, validation_data=(devX, devy_service))
    model.save(f'model_{model_type}.h5')

    # Predict service model on a subset of train and dev dataset
    model_type = "service"
    # load the model
    model = load_model(f'model_{model_type}.h5')
    recs = 5
    for x_in, y_in in [('trainX', 'y_train_service'), ('devX', 'y_dev_service')]:
        preds = model.predict(eval(x_in)[:recs], verbose=0)
        y = np.argmax(preds, axis=-1)
        ln = '\nModel: %s/%s\n\tRecs used: %d\n\tGiven Labels: %s\n\tPredicted Labels: %s\n' % (model_type, x_in, recs, ec.label_num_to_str(eval(y_in)[:recs], total_service_y), ec.label_num_to_str(y, total_service_y))
        print(ln)
    # Predict model on test dataset
    preds = model.predict(testX, verbose=0)
    y = np.argmax(preds, axis=-1)
    service_preds = ec.label_num_to_str(y, total_service_y)

    # predict crisp classes for test set
    yhat_classes = model.predict_classes(testX, verbose=0)
    # f1_service = f1_score(preds, yhat_classes)

    # define manual groups  model
    model_type = 'manual'
    model = ec.define_model(vocab_size, n_manual_classes, n_filters, n_kernal_size, n_dropout, optim)
    history = model.fit(trainX, trainy_manual, verbose=1, batch_size=n_batch_size, epochs=n_epochs, validation_data=(devX, devy_manual))
    model.save(f'model_{model_type}.h5')

    # Predict service model on a subset of train and dev dataset
    model_type = "manual"
    # load the model
    model = load_model(f'model_{model_type}.h5')
    recs = 5
    for x_in, y_in in [('trainX', 'y_train_manual'), ('devX', 'y_dev_manual')]:
        preds = model.predict(eval(x_in)[:recs], verbose=0)
        y = np.argmax(preds, axis=-1)
        ln = 'Model: %s/%s, Recs used: %d, Given Labels: %s, Predicted Labels: %s\n' % (model_type, x_in, recs, ec.label_num_to_str(eval(y_in)[:recs], total_manual_y), ec.label_num_to_str(y, total_manual_y))
        print(ln)

    # Predict model on test dataset
    # In order to find out 4 closest manuals for a given email, we apply manualgroups model to test dataset and
    # obtain preds. This "preds" is a numpy array with probabilities for all manualgroups. We argsort this
    # array to obtain indices for 4 largest probabilities for which which we then find out manual groups names.
    preds = model.predict(testX, verbose=0)

    # predict crisp classes for test set
    yhat_classes = model.predict_classes(testX, verbose=0)
    # f1_manual = f1_score(preds, yhat_classes)

    # print("f1_score for services: ", f1_service)
    # print("f1_score for manual groups: ", f1_manual)

    # Find closest 4 manual groups for an email    
    y = np.argsort(preds)
    manual_preds = []
    for row in y:
        s_preds = ec.label_num_to_str(row, total_manual_y)
        s_preds.reverse()
        m_preds = [i for i in s_preds if i != ''][:4]
        manual_preds.append("|".join(m_preds))

    # Create a pandas dataframe for test dataset with two new columns for service prediction and manual groups prediction.
    df = pd.read_csv(url_test, sep=";", encoding="utf8")
    dfs = pd.DataFrame(service_preds)
    dfm = pd.DataFrame(manual_preds)
    df['service_pred'] = dfs
    df['manual_pred'] = dfm
    df.to_csv('final.csv', sep=";", encoding="utf8")

