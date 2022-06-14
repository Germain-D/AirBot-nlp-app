from flask import Flask, request
from flask_cors import CORS, cross_origin
import pandas as pd
import contractions
import re
import nltk
import string
from nltk.tokenize import  word_tokenize
import numpy as np
from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim.corpora as corpora
from pprint import pprint# number of topics
import gensim
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import random

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

#preprocessing


def remove_punctuation(word_list):
    PUNCUATION_LIST = list(string.punctuation)
    return [w for w in word_list if w not in PUNCUATION_LIST]


def preproc_pipe(data):
    
    data = data.dropna()
    data_clean = pd.DataFrame()
    data_clean['text'] = data
    
    #text to lowercase
    data_clean['text'] = data_clean['text'].str.lower()

    #remove URL links
    data_clean['text'] = data_clean['text'].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
    data_clean['text'].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))

    #remove placeholders
    data_clean['text'] = data_clean['text'].apply(lambda x: re.sub(r'{link}', '', x))
    data_clean['text'] = data_clean['text'].apply(lambda x: re.sub(r"\[video\]", '', x))

    #remove HTML reference characters
    data_clean['text'] = data_clean['text'].apply(lambda x: re.sub(r'&[a-z]+;', '', x))

    #remove handles
    data_clean['text'] = data_clean['text'].apply(lambda x: re.sub(r"@([a-zA-Z0-9_]{1,50})","", x))

    #remove non-letter characters
    data_clean['text'] = data_clean['text'].apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', x))

    # Replace contractions with their longer forms 
    data_clean['text'] = data_clean['text'].apply(lambda x:  contractions.fix(x))



    #tokenize
 

    data_clean['tokens'] = data_clean['text'].apply(nltk.word_tokenize)

    #remove punctuation
    data_clean['tokens'] = data_clean['tokens'].apply(remove_punctuation)


   
    #remove stopwords
    stop_words = list(get_stop_words('en'))         #About 900 stopwords
    nltk_words = list(stopwords.words('english')) #About 150 stopwords
    stop_words.extend(nltk_words)

    data_clean['tokens'] = data_clean['tokens'].apply(lambda x: [w for w in x if w not in stop_words])

    #stemming/lemmatization

      
    lemmatizer = WordNetLemmatizer()
    data_clean['tokens'] = data_clean['tokens'].apply(lambda x: [lemmatizer.lemmatize(w) for w in x])

    #remove non english word
    #words = set(nltk.corpus.words.words())
    #data_clean['tokens'] = data_clean['tokens'].apply(lambda x: [w for w in x if w in words])


    return data_clean




def vectorize(tokenized_sentence,model):
    result = []
    for token in tokenized_sentence:
        if(token in model.key_to_index):
            result.append(model[token])
    return np.mean(result, axis=0)



def reponse (sentence,model,df):
    #put sentence to a dataframe
    d = {'text': [sentence]}
    phrase = pd.DataFrame(data=d)
    phrase['text']

    #just tokenize the sentence
    phrase['tokens'] = phrase['text'].apply(nltk.word_tokenize)
    phrase['tokens']

    #and vectorized it
    phrase['vectorized'] = phrase['tokens'].apply(vectorize,args=(model,))
    phrase['vectorized']
    
    compt = 0
    cosi = []
    for i in df.index:
        cosi.append(float(cosine_similarity(phrase['vectorized'][0].reshape(1, -1),df['question_vect'][i].reshape(1, -1))))
        compt +=1
    df['cosine'] = cosi

    return df[['responce','question','cosine']].nlargest(1, ['cosine'])['responce'].values[0]




def bot(model,df,sentence):
    flag=True

    while(flag==True):
        user_response = sentence
        print("You : "+user_response)
        user_response=user_response.lower()
        if(user_response!='bye'):
            if(user_response=='thanks' or user_response=='thank you' ):
                flag=False
                
                return "You are welcome.."
            else:
                if(greeting(user_response)!=None):
                    
                    return greeting(user_response)
                else:
                    
                    return reponse(user_response,model,df)
                    #sent_tokens.remove(user_response)
        else:
            flag=False
            
            return "Bye! take care.."

def greeting(sentence):
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
    GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)



def main(sentence):
    print('Début....')
    df = pd.read_csv('question_responce.csv')
    print('csv chargé....')	
    df[['responce_clean','token_responce']] = preproc_pipe(df['responce'])
    df[['question_clean','token_question']] = preproc_pipe(df['question'])
    print('data propre....')
    model = KeyedVectors.load_word2vec_format('glove.twitter.27B.100d.word2vec', binary=False)
    print('model chargé...')

    df['responce_vect'] = df['token_responce'].apply(vectorize,args=(model,))
    df['question_vect'] = df['token_question'].apply(vectorize,args=(model,))
    df=df.dropna()
    print('vecteurs créé....')
    print("AirBOT: My name is AirBOT. I will answer your queries about our flights. If you want to exit, type Bye!")
    return bot(model,df, sentence)




app = Flask(__name__)
CORS(app)
@app.route('/')
@cross_origin()
def index():
    args = request.args
    message = args.get('message')
    print("SENTENCE:", message)
    tmp = main(message).split(" ")[1:]
    return (" ").join(tmp)


if __name__ == '__main__':
    app.run(port=5000,debug=True)