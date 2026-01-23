import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse

nltk.download('punkt_tab')      
nltk.download('wordnet')    
nltk.download('omw-1.4') 
nltk.download('averaged_perceptron_tagger_eng')


DELIMS = ['.', '-', '?', '=', '/', '%20', '+', ':', '_', '~']

FILEPATH = '/var/my-data/datasets/harisudhan411/phishing-and-legitimate-urls/new_data_urls.csv'

raw_data = pd.read_csv(FILEPATH)


def _has_domain(url: str) -> bool:
    if 'http' not in url:
        url = 'http://' + url

    url_object = urlparse(url)
    res = re.findall(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', url_object.netloc)

    if res:
        return False
    return True


def _token_split(token: str, delim_idx: int = 0) -> list[str]:
    results = []

    if delim_idx == len(DELIMS):
        return [token]

    tokens = token.split(DELIMS[delim_idx])
    for t in tokens:
        if t:
            results.extend(_token_split(t, delim_idx + 1))
        else:
            continue

    return results


def get_url_features(url: str) -> list[bool | int]:

    domain = _has_domain(url)
    length = len(url)
    n_dots = url.count('.')
    n_underscores = url.count('_')
    n_dashes = url.count('-')
    n_numbers = len(re.findall(r'\d', url))

    return [domain, length, n_dots, n_underscores, n_dashes, n_numbers]


def tokenize_url(url: str) -> list[str]:
    tokens = []

    if 'http' not in url:
        url = 'http://' + url
    url = url.lower()

    o = urlparse(url)

    tokens.append(o.scheme.strip('://'))

    tokens.extend(_token_split(o.netloc))
    tokens.extend(_token_split(o.path))
    tokens.extend(_token_split(o.params))
    tokens.extend(_token_split(o.query))
    tokens.extend(_token_split(o.fragment))

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    
    return lemmatized_words


def main():
    i = 0
    for url in raw_data['url']:
        tokens = tokenize_url(url)
        print(tokens)
        if i == 50:
            break
        i += 1


if __name__ == '__main__':
    main()
