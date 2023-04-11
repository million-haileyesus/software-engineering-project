import numpy as np
import nltk
import re
import string 
import random
import t5
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


with open("data.txt", "r", encoding = "UTF-8") as data_file:
    text = data_file.read()
    text = text.lower()

def LemNormalize():      
    remove_punctuations = dict((ord(punct), None) for punct in string.punctuation)        
    lem = nltk.stem.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower().translate(remove_punctuations))
    return [lem.lemmatize(token) for token in tokens]

def greetings(greeting_sentence):
    greeting_inputs = ["hello", "hi", "hey", "how is it going?", "how are you doing?", "what's up", "whats up"]
    greeting_response = ["Hello", "Hi", "Hey", "How is it going?", "How are you doing?"]
        
    if greeting_sentence in greeting_inputs:
        return random.choice(greeting_response)

def response(user_response):
    sentence_tokens = nltk.sent_tokenize(text)
    word_tokens = nltk.word_tokenize(text)
    bot_response = ""
    TfidVec = TfidfVectorizer(tokenizer= lambda x : LemNormalize(), stop_words="english")
    tfidf = TfidVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1:], tfidf)[0]
    idx = vals.argsort()[-2]
    req_tfidf = vals[idx]    

    if(req_tfidf == 0):
        bot_response = bot_response + "I am unable to answer that question, sorry."
    else:
        bot_response = bot_response + sentence_tokens[idx] + " "
    
    return bot_response

def chat_flow():
    sentence_tokens = nltk.sent_tokenize(text)
    word_tokens = nltk.word_tokenize(text)

    flag = True
    print("Bot: Hello! I am Unibot. Please type any questions you might have")
    
    while(flag):
        print("User: ")
        user_response = input()
        user_response = user_response.lower()

        if(user_response == "bye"):
            flag = False
            print("Bot: Goodbye!")

        else:
            if(user_response.startswith("Thank".lower())):
               flag = False
               print("Bot: You are Welcome")
            
            else:
                if(greetings(user_response) != None):
                    print("Bot: " + greetings(user_response))
               
                else:
                    sentence_tokens.append(user_response)
                    word_tokens = word_tokens + nltk.word_tokenize(user_response)
                    final_words = list(set(word_tokens))
                    print("Bot: ", end = "")
                    print(response(final_words))
                    sentence_tokens.remove(user_response)

chat_flow()
