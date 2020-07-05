
from random import sample
import csv
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans



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

    def sampling(self, regex_file):
        read_csv = csv.reader(regex_file, delimiter = ',')
        tweet_list = []
        for row in read_csv:
            tweet = row[0]
            tweet_list.append(tweet)
        self.sampled_tweets = sample(tweet_list, 10)
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


    def zip_file(self, pattern, name):
        if request.method == 'POST':
            if os.path.isdir(self.path):      #making zip file for the obtained clusters
                    tweets_file = io.BytesIO()
                    with zipfile.ZipFile(tweets_file, 'w', zipfile.ZIP_DEFLATED) as my_zip:
                        for root, dirs, files in os.walk(self.path):
                            for file in files:
                                if re.search(pattern, file):
                                    my_zip.write(os.path.join(root, file))
                    tweets_file.seek(0)
        return send_file(tweets_file, mimetype='application/zip', as_attachment=True, attachment_filename=name+".zip")
