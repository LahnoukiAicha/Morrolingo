from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import math
import re
import random
import string
import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def readData(lang1, lang2, dataPath, reverse = False):
    print("Reading dataset...")
    
    #open the file and split by lines (\n)
    lines = open('C:/Users/Aicha LAHNOUKI/Desktop/flask/test.txt', encoding = 'utf-8').read().strip().split('\n')
    
    #split lines into pairs (separated by tab, or \t) and normalize
    
    pairs = [[cleanString(s) for s in l.split('\t')] for l in lines]
    
    # Reverse if spesified
    
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        source_lang = LangDict(lang2)
        target_lang = LangDict(lang1)
    else:
        source_lang = LangDict(lang1)
        target_lang = LangDict(lang2)
    
    return source_lang, target_lang, pairs


SOS_token = 0
EOS_token =1

class LangDict:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS", 1:"EOS"}
        self.n_words = 2 # SOS + COS = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
            
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def cleanString(s):
    #transform letters to lower case ones and remove non-letter symbols
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

Max_length = 10

exn_prefix = ("i am", "i m",
              "he is", "he s",
              "she is", "she s",
              "you are", "you re",
              "we are", "we re",
              "they are", "they re"              
             )

def filterPair(p):
    p1_tok = p[0].split(' ')
    p2_tok = p[1].split(' ')
    
    if len(p1_tok) < Max_length and len(p2_tok) < Max_length:
        return True
    else:
        return False

def BuildfilterdPairs(pairs):
    pairList = list()
    for pair in pairs:
        if filterPair(pair)==True:
            pairList.append([pair[0], pair[1]])
    return pairList

def prepareData(lang1, lang2, dataPath, reverse = False):
    input_lang, output_lang, pairs = readData(lang1, lang2, dataPath, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = BuildfilterdPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'darj', r'C:/Users/Aicha LAHNOUKI/Desktop/flask/test.txt', False)
print("Below is an example of sentence pair:")
print(random.choice(pairs))        

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device = device)
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim =1)

    
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device = device)
    

def indexFromSentence(lang, sentence):
    return [lang.word2index[word]  for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype = torch.long, device = device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


teacherForcing_r = 0.5

#one training iteration
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, 
         decoder_optimizer, criterion, max_length = Max_length):
    
    #initialize encoder hidden layer weights
    encoder_hidden = encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device = device)
    
    loss = 0
    
    #encoder
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    
    teacherForcing = True if random.random() < teacherForcing_r else False
    #decoder    
    if teacherForcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
            
            
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
        
            loss += criterion(decoder_output, target_tensor[di])
        
            if decoder_input.item() == EOS_token:
                break;
    
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s/ 60)
    s -= m * 60
    return '%dm %ds' % (m,s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


#iterate over training process
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every = 100, learning_rate = 0.01):
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                     for i in range(n_iters)]

    criterion = nn.NLLLoss()
    
    for iter in range(1, n_iters + 1):
        
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        
        loss = train(input_tensor, target_tensor, encoder,
                    decoder, encoder_optimizer, decoder_optimizer, criterion)
        
        print_loss_total += loss
        plot_loss_total += loss
        
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter/ n_iters), 
                                         iter, iter/ n_iters  *  100, print_loss_avg))
            
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
    showPlot(plot_losses)
            
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence, max_length=Max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device = device)
        for ei in range(input_length):

            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
            
            
        decoder_input = torch.tensor([[SOS_token]], device = device)#Start of sentens(SOS)
        
        decoder_hidden  = encoder_hidden
        
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi, = decoder_output.data.topk(1)
            
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break;
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            
            decoder_input = topi.squeeze().detach()
        
        return decoded_words, decoder_attentions[:di + 1]
    

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)

trainIters(encoder1, decoder1, 60000, print_every=5000)


"""def evaluateRandomely(encoder, decoder,):
        inpute=input('Sentence from source language: ' )
        output_words, attentions = evaluate(encoder, decoder, inpute)
        output_sentence = ' '.join(output_words)
        print('Model generated sentence: ', output_sentence)
        print('')"""
