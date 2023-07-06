import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tensorflow
import random
import pickle
import json
import tflearn
import re
from word2number import w2n

with open("data.json")as file:
    data = json.load(file)

try:
    with open("textdata.pickle", "rb")as f:
        words, labels, training, output = pickle.load(f)

except:
    words = [] 
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["prompts"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds) 
            docs_y.append(intent["label"])

        if intent["label"] not in labels:
            labels.append(intent["label"])


    words = [stemmer.stem(w.lower()) for w in words if w != "?"]   
    words = sorted(list(set(words))) 

    labels = sorted(labels)
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for idx, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1) # words in wrds are compared to words
            else:
                bag.append(0) 

        output_row = out_empty[:]
        output_row[labels.index(docs_y[idx])] = 1
        
        training.append(bag)
        output.append(output_row)
        
    training = numpy.array(training)
    output = numpy.array(output)

    with open("textdata.pickle", "wb")as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)


try:
    model.load("nlp.tflearn")

except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("nlp.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat(text):
    inp = text.lower()
    num_from_prommpt = []
    split_parts = []
    lit = inp.split()
    current_part = ""

    tokens = re.findall(r'\b\d+\b', inp)  # Extract individual numerical tokens

    for token in tokens:
        try:
            number = int(token)
            num_from_prommpt.append(number)
        except ValueError:
            try:
                number = w2n.word_to_num(token)
                num_from_prommpt.append(number)
            except ValueError:
                pass

    for char in inp:
        if char == '.' or char == ',' or char == '?':
            split_parts.append(current_part.strip())
            current_part = ""
        
        elif 'and' in lit:
            ind = lit.index('and')
            f, s = lit[:ind], lit[ind+1:]
            f1, s1 = ' '.join(f), ' '.join(s)
            split_parts.append(f1)
            split_parts.append(s1)                         

        else:
            current_part += char

    split_parts.append(current_part.strip())

    results_list = []

    for inp in split_parts:
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        if results[results_index] > 0.7:
            tag = labels[results_index]
            results_list.append(tag)


    results_list = list(set(results_list))  

    return results_list, num_from_prommpt


