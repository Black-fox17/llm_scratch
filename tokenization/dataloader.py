from datasets import load_dataset
import unicodedata

def get_dataset():
    return load_dataset("roneneldan/TinyStories")

def clean_wikitext(text):
    text = text.replace("@-@", "-")

    text = unicodedata.normalize("NFKC", text)

    text = text.strip()

    return text

def create_corpus(dataset):
    train_corpus = []
    for entry in dataset['train'].take(10000):
        text = entry['text']
        if len(text.strip()) > 0:
            train_corpus.append(clean_wikitext(text))
    
    return train_corpus