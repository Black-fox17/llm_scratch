from datasets import load_dataset
import unicodedata

def get_dataset(dataset_name="roneneldan/TinyStories"):
    return load_dataset(dataset_name)

def clean_wikitext(text):
    text = text.replace("@-@", "-")

    text = unicodedata.normalize("NFKC", text)

    text = text.strip()

    return text

def create_corpus(dataset, num_samples=10000):
    train_corpus = []
    for entry in dataset['train'].take(num_samples):
        text = entry['text']
        if len(text.strip()) > 0:
            train_corpus.append(clean_wikitext(text))
    
    return train_corpus