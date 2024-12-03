import numpy as np
import nltk
nltk.download('punkt') # Download the Punkt sentence tokenizer
nltk.data.path.append('/home/mkh/nltk_data')
nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"



def tokenize_fn(text):
    """
    split test into list of words/tokens --> word , punctuation , number
    """
    tokenization = nltk.word_tokenize(text)
    return tokenization


def stem_fn( word ):
    """
    # to find the root of the words
    # playing , played, playes --> [ 'play' , 'play' , 'play' ]
    #["organization" , "organizer" , "organizing" ] ---> ["organ" , "organ", "organ"] 
    """
    stemmer = PorterStemmer()
    stemming = stemmer.stem(word.lower())
    return  stemming

def bag_of_words(tokenized_sentence,words):
    """
    return bag of words : 1 for each known word that exist in the test , 0 otherwise
    example : 
    sentence = ["hello" , "how", "are" ,"you"]
    words = ["hi" , "hello", "I" , "bye", "thank", " cool"]
    bag  =  [ 0   ,  1    , 0    ,  0   , 0    , 0  ]
    """
    # stem each word 
    sentence_words = [stem_fn(word) for word in tokenized_sentence]
    
    # set the intial values in bag with 0 for each words
    bag = np.zeros(len(words) , dtype= np.float32)
    # print("Zeroooooooooooooooooooooooooooooo",bag)
    for idx, w in enumerate(words) : 
       if w in sentence_words:
            # print(idx,w)
            bag[idx] = 1
    
    return bag

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true , y_pred).sum().item()
    accuracy= (correct/len(y_pred))*100
    return accuracy
    
def train_fn(model:nn.Module,
             data_loader : torch.utils.data.DataLoader,
             optimizer :torch.optim.Adam,
             loss_fn : torch.nn.Module,
             accuracy_fn, 
             device: torch.device = device,
            prnt=True):
    trainLoss, trainAcc = 0 , 0
    model.to(device)

    model.train() # set model in the training mode

    for batch , (X,y) in enumerate(data_loader):
        X, y = X.to(device) , y.to(device)
        
        # 1- forward pass
        y_pred = model(X)
        # 2- CALCULATE THE LOSS
        loss = loss_fn(y_pred, y)
        trainLoss += loss
        trainAcc += accuracy_fn(y_true = y, y_pred =y_pred.argmax(dim =1))

        #3-OPTIMIZER zERO_Grad
        optimizer.zero_grad()

        # 4. loss backword
        loss.backward()
        #5. OPTIMIZER step
        optimizer.step()
    trainLoss/=len(data_loader)
    trainAcc /= len(data_loader)
    if prnt == True :
        print(f"model training loss is : {trainLoss: 0.5f} | model accuracy: {trainAcc}")
        
        






    