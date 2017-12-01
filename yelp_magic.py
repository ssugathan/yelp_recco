import json
import numpy as np
import csv
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

ngram_vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b')
ngram_analyze = ngram_vectorizer.build_analyzer()
porter = PorterStemmer()
include_stopwords = ['not', 'does', 'until', 'no', 'only', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
                     'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
                     'very']
omit_words = [stopword for stopword in stopwords.words('english') if stopword not in include_stopwords]
omit_chars = [';', '&', '|', '-', ']', '$', '%', '.', "'", '{', '\\', '(', '^', '/', '@', '`', ',', '"', '#', '_',
              ':', '~', ')', '+', '[', '>', '<', '=', '*', '?', '}']


def process(text):
    # convert to lowercase
    text = text.lower()
    # remove all special chars with spaces
    for c in omit_chars:
        if c in text:
            text = text.replace(c, " ")
    # replace exclamation mark with a fake word wrapped with spaces
    text = text.replace("!", " __exclamation__ ")
    # remove all stop words and stem, this code also rids of excess spaces
    text = ' '.join([porter.stem(word) for word in text.split() if word not in omit_words])
    return text

def sort_data(data):
    # by_labels = sorted(data, key=lambda review: review[1])
    index = int(len(data)*8/10)
    train = data[:index]
    test = data[index:]
    print ('There are ' + str(len(train)) + ' training samples and ' + str(len(test)) + ' testing samples')
    return train, test

def pretty_bag_of_words(bag, features_labels):
    plucked = {}
    for index, word_occurrence in enumerate(bag):
        if word_occurrence > 0:
            plucked[features_labels[index]] = word_occurrence
    return plucked

def bag_of_words(train, test):
    words_to_features = {}
    features_labels = []
    dictionary_count = {}
    train_bags = []
    test_bags = []

    for review in train:
        for word in review[0].split():
            dictionary_count[word] = 0 if word not in dictionary_count else dictionary_count[word] + 1

    feature_index = 0

    for word in dictionary_count:
        if dictionary_count[word] > 3:
            words_to_features[word] = feature_index
            features_labels.append(word)
            feature_index += 1

    bag_size = len(words_to_features)

    for review in train:
        bag = np.zeros(bag_size)
        for word in review[0].split():
            if word in words_to_features:
                bag[words_to_features[word]] += 1
        train_bags.append([bag, review[1]])

    for review in test:
        bag = np.zeros(bag_size)
        for word in review[0].split():
            if word in words_to_features:
                bag[words_to_features[word]] += 1
        test_bags.append([bag, review[1]])

    # print ('Features extracted as bags of words with dictionary size = ' + str(len(words_to_features)))
    # print ('train[0] = ' + str(pretty_bag_of_words(train_bags[0][0], features_labels)))
    # print ('train[1] = ' + str(pretty_bag_of_words(train_bags[1][0], features_labels)))
    return train_bags, test_bags, features_labels

def post_process(train_bags, test_bags, norm=0, scalar=1):
    for bag in train_bags + test_bags:
        norm_zero = np.linalg.norm(bag[0], ord=norm)
        bag[0] = [scalar * float(word_occurrence) / norm_zero if norm_zero != 0 else 0 for word_occurrence in bag[0]]
    # print ('Finished post processing normalizing l-' + str(norm) + ' norm')


def main_sentiment_prediction(train, test, features_labels=list(), naive_bayes=MultinomialNB):
    train_x, train_y, test_x, test_y = split_data_from_labels(train, test)
    return sentiment_prediction(train_x, train_y, test_x, test_y, features_labels, naive_bayes)


def sentiment_prediction(train_x, train_y, test_x, test_y, features_labels=list(), naive_bayes=MultinomialNB):
    logistic = LogisticRegression()
    logistic.fit(train_x, train_y)
    logistic_score = logistic.score(test_x, test_y)
    confusion_logistic = confusion_matrix(test_y, logistic.predict(test_x))
    positive_list = []

    # nb = naive_bayes()
    # nb.fit(train_x, train_y)
    # nb_score = nb.score(test_x, test_y)
    # confusion_nb = confusion_matrix(test_y, nb.predict(test_x))

    if len(features_labels) > 0:
        # Analyzing weights of features of logistic regression
        coefficients = [(coefficient, features_labels[feature_index]) for feature_index, coefficient in
                        enumerate(logistic.coef_[0])]
        coefficients = sorted(coefficients, key=lambda c_tuple: -abs(c_tuple[0]))
        positive_coefficients = sorted(coefficients, key=lambda c_tuple: -(c_tuple[0]))

        # Analyzing weights of features of naive bayes
        # nb_coefficients = [(coefficient, features_labels[feature_index]) for feature_index, coefficient in
        #                    enumerate(nb.coef_[0] - np.mean(nb.coef_[0]))]
        # nb_coefficients = sorted(nb_coefficients, key=lambda c_tuple: -abs(c_tuple[0]))
        # positive_nb_coefficients = sorted(nb_coefficients, key=lambda c_tuple: -(c_tuple[0]))

    print ('Logistic regression score = ', logistic_score)
    print ('Below the confusion matrix:')
    print (confusion_logistic)

    if len(features_labels) > 0:
        # print ('Below the words with the most significant role:')
        # print ([c_tuple[1] for c_tuple in coefficients[:5]])
        # print ('Below the words with the most significant positive role:')
        # print ([c_tuple[1] for c_tuple in positive_coefficients[:25]])
        positive_list = [c_tuple[1] for c_tuple in positive_coefficients[:25]]
    # print ('F: Naive Bayes score = ', nb_score)
    # print ('Below the confusion matrix:')
    # print (confusion_nb)

    # if len(features_labels) > 0:
    #     print ('Below the words with the most significant role:')
    #     print ([c_tuple[1] for c_tuple in nb_coefficients[:5]])
    #     print ('Below the words with the most significant positive role:')
    #     print ([c_tuple[1] for c_tuple in positive_nb_coefficients[:25]])

    return positive_list

def split_data_from_labels(train, test):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for sample in train:
        train_x.append(sample[0])
        train_y.append(sample[1])

    for sample in test:
        test_x.append(sample[0])
        test_y.append(sample[1])

    return train_x, train_y, test_x, test_y


def main_func(start, end):
    business_detailes_dict = {}
    reviews_dict = {}
    full_business_ids = [
    '4JNXUYY8wbaaDmk3BPzlWw','RESDUcs7fIiihp38-d6_6g','K7lWdNUhCbcnEvI0NhGewg','cYwJA2A6I12KNkm2rtXd5g','DkYS3arLOhA8si5uUEmHOw','f4x1YBxkLrZg652xt2KR5g','eoHdUeQDNgQ6WYEnP2aiRw','2weQS-RnoOBhb1KsHKyoSQ','KskYqH1Bi7Z_61pH6Om8pg','ujHiaprwCQ5ewziu0Vi9rw','iCQpiavjjPzJ5_3gPD5Ebg','rcaPajgKOJC2vo_l3xa42A','El4FC8jcawUVgw_0EIcbaQ','hihud--QRriCYZw1zZvW4g','g8OnV26ywJlZpezdBnOWUQ','7sPNbCx7vGAaH7SbNPZ6oA','XZbuPXdyA0ZtTu3AzqtQhg','OETh78qcgDltvHULowwhJg','P7pxQFqr7yBKMMI2J51udw','XXW_OFaYQkkGOGniujZFHg','yfxDa8RFOvJPQh0rNtakHA','HhVmDybpU7L50Kb5A0jXTg','Cni2l-VKG_pdospJ6xliXQ','QJatAcxYgK1Zp9BRZMAx7g','NvKNe9DnQavC9GstglcBJQ','3kdSl5mo9dWC4clrQjEDGg','YJ8ljUhLsz6CtT_2ORNFmg','RwMLuOkImBIqqYj4SSKSPg','fL-b760btOaGa85OJ9ut3w','UPIYuRaZvknINOd1w8kqRQ','G-5kEa6E6PD5fkBRuA7k9Q','ECOkEVUodMLUxvI0PMI4gQ','ZkGDCVKSdf8m76cnnalL-A','xkVMIk_Vqh17f48ZQ_6b0w','awI4hHMfa7H0Xf0-ChU5hg','pH0BLkL4cbxKzu471VZnuA','LNGBEEelQx4zbfWnlc66cw','pSQFynH1VxkfSmehRXlZWw','faPVqws-x-5k2CQKDNtHxw','JzOp695tclcNCNMuBl7oxA','I6EDDi4-Eq_XlFghcDCUhw','EAwh1OmG6t6p3nRaZOW_AA','LR0qF0FEVsCOhYWUOiH26A','l_GV0hgEoTUf70uJVT0_hg','0W4lkclzZThpx3V65bVgig','JpgVl3d20CMRNjf1DVnzGA','VyVIneSU7XAWgMBllI6LnQ','OgJ0KxwJcJ9R5bUK0ixCbg','QJR4qBUHegWEozSQrGmBPw','igHYkXZMLAc9UdV5VnR_AA','ugLqbAvBdRDc-gS4hpslXw','Fi-2ruy5x600SX4avnrFuA','3GEEy7RP6e4bT4LAiWFMFQ','JyxHvtj-syke7m9rbza7mA','9a3DrZvpYxVs3k_qwlCNSw','3l54GTr8-E3XPbIxnF_sAA','3BCsAgo_1i4xMuTyLKMLRQ','0FUtlsQrJI7LhqDPxLumEw','X8c23dur0ll2D9XTu-I8Qg','K-uQkfSUTwu5LIwPB4b_vg','frCxZS7lPhEnQRJ3UY6m7A','_w5hBpkjHs5_Hv3pLeHtIw','eLFfWcdb7VkqNyTONksHiQ','d10IxZPirVJlOSpdRZJczA','UNI1agsPX2k3eJSJVB91nw','wl0QZqAzr1DelslQ02JGCQ','xVEtGucSRLk5pxxN0t4i6g','VsewHMsfj1Mgsl2i_hio7w','2iTsRqUsPGRH1li1WVRvKQ','YPavuOh2XsnRbLfl0DH2lQ','wUKzaS1MHg94RGM6z8u9mw','L2p0vO3fsS2LC6hhQo3CzA','--9e1ONYQuAa-CB_Rrw7Tw','N0apJkxIem2E8irTBRKnHw','JDZ6_yycNQFTpUZzLIKHUg','FLMxWQO-ckCQmGZhU9OQgw','yNPh5SO-7wr8HPpVCDPbXQ','RJNAeNA-209sctUO0dmwuA','JLbgvGM4FXh9zNP4O5ZWjQ','j5nPiTwWEFr-VsePew7Sjg','dn_ipqbm7_jUz5X3rDez_A','d_L-rfS1vT3JMzgCUGtiow','fQt4D34vcJNtEf8Q4zte3w','GJ_bXUPv672YwNg4TneJog','5shgJB7a-2_gdnzc0gsOtg','kRgAf6j2y1eR0wOFdzFAuw','mDR12Hafvr84ctpsV6YLag','z6-reuC5BYf_Rth9gMBfgQ','rTS8LsUmNIiXsXydE49tPA','4GXII-GU7S0ZyU6ElkhscQ','CiYLq33nAyghFkUR15pP-Q','cHdJXLlKNWixBXpDwEGb_A','r_BrIgzYcwo1NAuG9dLbpg','gG9z6zr_49LocyCTvSFg0w','H8qpFitIesth86zqR4cwYg','cJWbbvGmyhFiBpG_5hf5LA','RAh9WCQAuocM7hYM5_6tnw','5T6kFKFycym_GkhgOiysIw','bpRo8L8dkhgbJhdIKa9mwA','sqRX-XLlhx4rs2c1TpBf8A','A-uZAD4zP3rRxb44WUGV5w','Xg5qEQiB-7L6kGJ5F4K3bQ','7fxebHYUwIF6CakxSr70iQ','OVTZNSkSfbl3gVB9XQIJfw','J4CATH00YZrq8Bne2S4_cw','IT_4EEIbv6Ox1jBRMyE7pg','GI-CAiZ_Gg3h21PwrANB4Q','LFs5jyYdXlzi0SpAYi1eSA','eS29S_06lvsDW04wVrIVxg','u-SJ5QUwrNquL9VnXwl8cg','pHJu8tj3sI8eC5aIHLFEfQ','aLcFhMe6DDJ430zelCpd2A','EnCIojgP5KTr1leaysFE3A','UUGoM4q4i8rK2CBRS0xDAw','A5Rkh7UymKm0_Rxm9K2PJw','zdE82PiD6wquvjYLyhOJNA','PXShA3JZMXr2mEH3on5clw','3N9U549Zse8UP-MwKZAjAQ','jWv5GUtEp30OD5L5C8c2DQ','0NmTwqYEQiKErDv4a55obg','R_ZlcX46pPdjhjmfd043LA','C8D_GU9cDDjbOJfCaGXxDQ','4k3RlMAMd46DZ_JyZU0lMg','sNVGdeOPeitJ3OWUQBINzQ','IMLrj2klosTFvPRLv56cng','uuGL8diLlHfeUeFuod3F-w','IsoLzudHC50oJLiEWpwV-w','RVQE2Z2uky4c0-njFQO66g','C9ImzBi5fn742ZcAYDww2A','RtUvSWO_UZ8V3Wpj0n077w','0AQnRQw34IQW9-1gJkYnMA','u4sTiCzVeIHZY8OlaL346Q','gTlDDzDEHyDQ6iwjNhpI6A','K0j_Znzin0jShXVnpvW86A','SVGApDPNdpFlEjwRQThCxA','nUpz0YiBsOK7ff9k3vUJ3A','whAwdYVty-jSNRhrYT2zHA','Gaq3S9lmjXVcuDCZ8ulppw','vl2IZrNJEA8npSjqXbdwxw','Jt28TYWanzKrJYYr0Tf1MQ','YhCAJ8acd1X7GkCHPhD8Xw','uW6UHfONAmm8QttPkbMewQ','mD7zqv7Y3kvsa_p_MtTayg','pKk7jCFIm96qDdk0laVT2w','PVTfzxu7of57zo1jZwEzkg','SAIrNOB4PtDA4gziNCucwg','fHM09_y3QX3n4a_bIFbk_w','k1QpHAkzKTrFYfk6u--VgQ','qqs7LP4TXAoOrSlaKRfz3A','lKom12WnYEjH5FFemK3M1Q','ZibmYdOPKLlqDM9oR6xzOA','gBfPyzPRmeOaj3SdcIj0Rw','aT_SsfZ6GQgJGyuIv1Hapw','LwQB9H3jZ9wTk24Lr-AnZQ','IhNASEZ3XnBHmuuVnWdIwA','utIA0LyQmwP-9DRyxUe6qQ','oXoVJ0xKv82cBo9U6oEjlQ'
    ]

    print("*** 1 ***")
    for business_id in full_business_ids[start:end]:
        reviews_dict[business_id] = []
        for line in open('business.json'):
            business_json = json.loads(line)
            if business_id == business_json['business_id']:
                business_detailes_dict[business_id] = [business_json['name'],business_json['city']]

    print("*** 2 ***")
    for line in open('review.json'):
        review_json = json.loads(line)
        business_id = review_json['business_id']
        if business_id in full_business_ids[start:end]:
            modified_text = process(review_json['text'])
            sentiment = 0
            if review_json['stars'] > 4:
                sentiment = 1
            temp_review = [modified_text, sentiment]
            reviews_dict[business_id].append(temp_review)

    print("*** 3 ***")
    positive_words_dict = {}
    for business_id in full_business_ids[start:end]:
        print('========================= ', business_id, ' =========================')
        train, test = sort_data(reviews_dict[business_id])
        train_bags, test_bags, features_labels = bag_of_words(train, test)
        post_process(train_bags, test_bags, norm=0, scalar=2*np.pi)
        positive_words_dict[business_id] = main_sentiment_prediction(train_bags, test_bags, features_labels)
        print(positive_words_dict[business_id])

    print("*** 4 ***")
    csv_data = []
    for business_id in full_business_ids[start:end]:
        csv_data.append([business_id, business_detailes_dict[business_id][0], business_detailes_dict[business_id][1], positive_words_dict[business_id]])
    file_name = 'processed_data_' + str(start) + '_' + str(end) + '.csv'
    myFile = open(file_name, 'w')
    with myFile:
       writer = csv.writer(myFile)
       writer.writerows(csv_data)

main_func(0, 20)
main_func(20, 40)
main_func(40, 60)
main_func(60, 80)
main_func(80, 100)
main_func(100, 120)
main_func(120, 140)
main_func(140, 157)
