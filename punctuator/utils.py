# coding: utf-8
import numpy as np  

def get_reverse_map(dictionary):
    return {v:k for k,v in dictionary.items()}

def get_vocabulary_size(vocabulary):
    return max(vocabulary.values()) + 1

def input_word_index(vocabulary, input_word):
    return vocabulary.get(input_word, vocabulary["<unk>"])

def punctuation_index(punctuations, punctuation):
    return punctuations[punctuation]

def load_vocabulary(file_path):
    with open(file_path, 'r') as vocab:
        vocabulary = {w.strip(): i for (i, w) in enumerate(vocab)}
    if "<unk>" not in vocabulary:
        vocabulary["<unk>"] = len(vocabulary)
    if "<END>" not in vocabulary:
        vocabulary["<END>"] = len(vocabulary)
    return vocabulary
    
def load_model(file_path):
    from . import models
    
    model = np.load(file_path)
    net = getattr(models, model["type"])()
    
    net.load(model)
    
    return net

def prepare_for_punctuate(model_name):
    # pre-load the large models once
    net = load_model(model_name)
    net.batch_size = 1
    net.reset_state()
    punctuation_reverse_map = get_reverse_map(net.out_vocabulary)

    return (net, punctuation_reverse_map)

def punctuate(unpunctuated_text, net, punctuation_reverse_map):
    # run the prediction as many times as needed
    net.reset_state()
    punctuated_text = ""

    stream = unpunctuated_text.split()
    for word in stream:

        word_index = input_word_index(net.in_vocabulary, word)
        punctuation_index = net.predict_punctuation([word_index], np.array([0.0]))[0]

        punctuation = punctuation_reverse_map[punctuation_index]

        if punctuation == " ":
            punctuated_text = punctuated_text + ("%s%s" % (punctuation, word))
        else:   
            punctuated_text = punctuated_text + ("%s %s" % (punctuation[:1], word))

    return punctuated_text