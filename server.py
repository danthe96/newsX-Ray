from flask import Flask, render_template, request, jsonify
import os
import json
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import subprocess

app = Flask(__name__)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

SIMILARITY_THRESHOLD = 0.861


port = int(os.getenv('PORT', 8000))
english_stopwords = set(stopwords.words('english'))


def clean_text(text):
    # Sentences are preprocessed and simplified for analysis,
    # but full sentences should be returned to the Chrome extension
    original_mapping = {}

    text = text.replace('<p>', '').replace('</p>', '').replace('<p/>', '')
    sentences = tokenizer.tokenize(text)
    basic_sentences = []
    for sentence in sentences:
        sentence2 = sentence.replace('\\n', ' ').replace('\\"', '\'').replace('\"', '\'')
        sentence2 = ' '.join([i for i in sentence2.lower().split() if i not in english_stopword])
        basic_sentences.append(sentence2)
        original_mapping[sentence2] = sentence

    return (''.join(['\"' + s + '\"\n' for s in basic_sentences]), original_mapping)


def calculate_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


@app.route('/', methods=['POST'])
def check_similarity():
    print(request)
    reuters_text, original_r_s = clean_text(request.json['reuters'])
    news_text, original_n_s = clean_text(request.json['news'])

    reuters_id = request.json['reuters_id']
    newssource = request.json['source']

    if os.path.isfile('cache/' + reuters_id + '_' + newssource + '_vec1.dat'):
        reuters_data = pd.read_pickle('cache/' + reuters_id + '_' + newssource + '_vec1.dat')
        news_data = pd.read_pickle('cache/' + reuters_id + '_' + newssource + '_vec2.dat')
    else:
        cutoff = len(reuters_text.split('\n'))
        f = open('text_temp', 'w')
        f.write(reuters_text)
        f.write(news_text)
        f.close()
        # Call fasttext with pretrained model. Short solution that loads from SSD everytime, but should be used with
        # Popen.communicate in non-Hackathon environment
        p = subprocess.Popen('fasttext/fasttext print-sentence-vectors fasttext/models/wiki.en.bin < text_temp > temp_result',
                             shell=True)
        p.wait()

        result = pd.read_csv('temp_result', sep='\s+', quotechar='"', index_col=0, header=None)
        reuters_data = result.iloc[:(cutoff - 1)]
        news_data = result.iloc[cutoff:]
        reuters_data.to_pickle('cache/' + reuters_id + '_' + newssource + '_vec1.dat')
        news_data.to_pickle('cache/' + reuters_id + '_' + newssource + '_vec2.dat')

    reuters_sentences = reuters_data.index
    news_sentences = news_data.index

    info_in_both = []
    for n_s in news_sentences:
        for r_s in reuters_sentences:
            similarity = calculate_similarity(reuters_data.loc[r_s], news_data.loc[n_s])

            if similarity > SIMILARITY_THRESHOLD:
                info_in_both.append({
                    'news_sentence': original_n_s[n_s],  # Sentence by news outlet
                    'reuters_sentence': original_r_s[r_s],  # Sentence by news agency
                    'score': similarity})

    # Sentences most likely written by news outlet itself
    info_in_news = [original_n_s[n_s] for n_s in news_sentences
                    if not any(x for x in info_in_both if x['news_sentence'] == original_n_s[n_s])]
    # Sentences omitted from news agency report
    info_in_reuters = [original_r_s[r_s] for r_s in reuters_sentences
                       if not any(x for x in info_in_both if x['reuters_sentence'] == original_r_s[r_s])]

    return jsonify({
        'matched_sentences': info_in_both,
        'news_additions': info_in_news,
        'omitted_sentences': info_in_reuters
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
