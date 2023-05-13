import nltk
import string
import warnings
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from nltk import porter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from spacy import load


nlp = load("en_core_web_sm")
stopwords_list = stopwords.words('english')
warnings.filterwarnings("ignore")

def queries():
    with open("data.txt", "r", encoding = "UTF-8") as data_file:
        data = data_file.readlines()
        data = [re.sub(r'[\n\t\r]+', ' ', line) for line in data]
    data_file.close()

    return data

def preprocess_text(input_text):
    lem = WordNetLemmatizer()
    remove_punctuations = dict((ord(punct), None) for punct in string.punctuation)
    tokens = nltk.word_tokenize(input_text)
    filtered_tokens = [lem.lemmatize(token) for token in tokens]
    filtered_tokens = [token.translate(remove_punctuations) for token in filtered_tokens]

    return filtered_tokens

text = queries()
sentence_tokens = text

def greetings(greeting_sentence):
    greeting_inputs = ["hello", "hi", "hey", "how is it going?", "how are you doing?", "what's up", "whats up", "hi there"]
    greeting_response = ["Hello", "Hi", "Hey", "Howdy", "How is it going?", "How are you doing?"]

    if greeting_sentence in greeting_inputs:
        return random.choices(greeting_response)[0]

def response(user_response):
    bot_response = ""
    similar_scores = train(user_response)

    if not similar_scores:
        bot_response = bot_response + "I am unable to answer that question, sorry."

    else:
        score = max(similar_scores)

        if score < 0.80:
            bot_response = bot_response + "I am unable to answer that question, sorry.\
                                            \nCould you be more specific?"

        else:
            idx = similar_scores.index(score)
            bot_response = bot_response + text[idx].strip().capitalize()

    return bot_response

def train(user_response):
    tfidVec = TfidfVectorizer(tokenizer=preprocess_text, stop_words=stopwords_list)
    tfidf = tfidVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf).flatten()
    similarity_scores = []

    for i in range(len(sentence_tokens) - 1):
        ratio = fuzz.token_set_ratio(sentence_tokens[i], user_response)
        weighted_score = vals[i] + (ratio / 100)
        similarity_scores.append(weighted_score)

    return similarity_scores

def chat_flow():
    bot_response = ""
    flag = True
    print("Bot: Hi there! How can I assist you today?")

    while flag:
        user_response = input("User: ")
        user_response = user_response.lower()

        if len(user_response) == 0:
            print("Bot: Please ask your question")

        elif "bye" in user_response:
            flag = False
            print("Bot: Goodbye!")

        else:
            if user_response.startswith("Thank".lower()):
                print("Bot: You are Welcome. Is that all")
                
            else:
                greet = greetings(user_response)
                if  != None:
                    print("Bot: " + greet)

                else:
                    sentence_tokens.append(user_response)
                    bot_response = f"Bot: {response(user_response)}"
                    sentence_tokens.remove(user_response)
                    print(bot_response)

chat_flow()
