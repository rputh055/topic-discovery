
from random import sample
import csv
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import zipfile
import io
from flask import send_file
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
import json


class Clustering(object):
    """docstring for Clustering."""
    def __init__(self, path):
        self.path = path
        self.tweet_embeddings = []
        self.sampled_tweets = []


    def regex(self, file):
        with open(os.path.join(self.path, file), 'r',) as infile, open(self.path + '/processed.csv', 'w') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            for row in reader:
                newrow = [re.sub(r'\'[0-9]*\', ', "", str(row))]
                newrow = [re.sub(r'[^0-9a-zA-Z.,? ]+', "", str(newrow))]
                writer.writerow(newrow)
        outfile.close()

    def sampling(self):
        with open(self.path+'/processed.csv') as csv_file:

            read_csv = csv.reader(csv_file, delimiter = ',')
            tweet_list = []

            for row in read_csv:
                tweet = row[0]
                tweet_list.append(tweet)
            self.sampled_tweets = sample(tweet_list, 99)
            return self.sampled_tweets

    def embeddings(self, list_of_tweets):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.tweet_embeddings = model.encode(list_of_tweets)
        return self.tweet_embeddings

    def cluster(self, k):
        k_means = KMeans(n_clusters = k)
        k_means.fit(self.tweet_embeddings)
        k_pred = k_means.predict(self.tweet_embeddings)
        cluster_df = pd.DataFrame({"text": self.sampled_tweets, "cluster" : k_pred})
        for i in range(k):
            df = cluster_df[cluster_df['cluster'] == i]
            df.drop(columns = ['cluster'], inplace = True)
            df.to_csv(self.path+"/cluster"+str(i)+".txt")

    def summaries(self, k):
        for i in range(k):
            with open(self.path +'/cluster'+str(i)+'.txt', 'r') as file:
                data = file.read().replace('\n', '')

                model = T5ForConditionalGeneration.from_pretrained('t5-small')
                tokenizer = T5Tokenizer.from_pretrained('t5-small')
                device = torch.device('cpu')

                preprocess_text = data.strip().replace("\n","")

                t5_prepared_Text = "summarize: "+preprocess_text
                #print ("original text preprocessed: \n", preprocess_text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

                # summmarize
                summary_ids = model.generate(tokenized_text,
                                                    num_beams=4,
                                                    no_repeat_ngram_size=2,
                                                    min_length=30,
                                                    max_length=100,
                                                    early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                #print ("\n\nSummarized text: \n",output)
                with open(self.path + '/summary'+str(i)+'.txt', 'w') as sum:
                    sum.write(output)



    def zip_file(self, pattern, name):
        if os.path.isdir(self.path):      #making zip file for the obtained clusters
            tweets_file = io.BytesIO()
            with zipfile.ZipFile(tweets_file, 'w', zipfile.ZIP_DEFLATED) as my_zip:
                for root, dirs, files in os.walk(self.path):
                    for file in files:
                        if re.search(pattern, file):
                            my_zip.write(os.path.join(root, file))
            tweets_file.seek(0)
        return send_file(tweets_file, mimetype='application/zip', as_attachment=True, attachment_filename=name+".zip")
