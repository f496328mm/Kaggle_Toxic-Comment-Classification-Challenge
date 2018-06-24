
# https://github.com/zake7749/DeepToxic/blob/master/clean_data.ipynb

import re
from nltk.stem import SnowballStemmer
from collections import defaultdict
from keras.preprocessing import text, sequence
#print('Processing text dataset')
# Regex to remove all Non-Alpha Numeric and space
special_character_removal=re.compile(r'[^?!.,:a-z\d ]',re.IGNORECASE)

# regex to replace all numerics
replace_numbers=re.compile(r'\d+',re.IGNORECASE)
word_count_dict = defaultdict(int)
clean_word_dict = {}

def clean_text(text, remove_stopwords=False, stem_words=False, count_null_words=True, clean_wiki_tokens=True):
    # Clean the text, with the option to remove stopwords and to stem words.
    # dirty words
    text = text.lower()
    text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", text)
    text = re.sub(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}", "", text)
    
    if clean_wiki_tokens:
        # Drop the image
        text = re.sub(r"image:[a-zA-Z0-9]*\.jpg", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.png", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.gif", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.bmp", " ", text)

        # Drop css
        text = re.sub(r"#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})", " ",text)
        text = re.sub(r"\{\|[^\}]*\|\}", " ", text)
        
        # Clean templates
        text = re.sub(r"\[?\[user:.*\]", " ", text)
        text = re.sub(r"\[?\[user:.*\|", " ", text)        
        text = re.sub(r"\[?\[wikipedia:.*\]", " ", text)
        text = re.sub(r"\[?\[wikipedia:.*\|", " ", text)
        text = re.sub(r"\[?\[special:.*\]", " ", text)
        text = re.sub(r"\[?\[special:.*\|", " ", text)
        text = re.sub(r"\[?\[category:.*\]", " ", text)
        text = re.sub(r"\[?\[category:.*\|", " ", text)
    
    for typo, correct in clean_word_dict.items():
        text = re.sub(typo, " " + correct + " ", text)

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = replace_numbers.sub(' ', text)
    text = special_character_removal.sub('',text)

    if count_null_words:
        text = text.split()
        for t in text:
            word_count_dict[t] += 1
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return (text)

#train_x = train["comment_text"].fillna("no comment").values
#test_x = test["comment_text"].fillna("no comment").values

#train_x = [clean_text(text) for text in train_x]    
#test_x = [clean_text(text) for text in test_x]

def data2vector(train,test,maxlen,max_features):
    #------------------------------------------
    print('clean_text')
    train_x = train["comment_text"].fillna("no comment").values
    test_x = test["comment_text"].fillna("no comment").values
    
    train_x = [clean_text(text) for text in train_x]    
    test_x = [clean_text(text) for text in test_x]
    #train_x = train["comment_text"].values
    #test_x = test["comment_text"].values
    labels = list( train.columns )
    [ labels.remove(rm) for rm in ['comment_text','id'] ]
    train_y = train[labels].values
    #-----------------------------------------------------------
    print('build dict by numwords')
    tokenizer = text.Tokenizer(num_words = max_features)
    tokenizer.fit_on_texts(list(train_x))# build dict by numwords
    
    seq_train_x = tokenizer.texts_to_sequences(train_x)# text to seq
    seq_test_x = tokenizer.texts_to_sequences(test_x)
    vector_train_x = sequence.pad_sequences(seq_train_x, maxlen=maxlen)# text to vector
    vector_test_x = sequence.pad_sequences(seq_test_x, maxlen=maxlen)# text to vector
    
    return vector_train_x,train_y,vector_test_x,tokenizer,labels





