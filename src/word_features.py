from gensim.models import FastText
from transformers import BertTokenizer, BertModel
from label import Label_Ref
import torch
import numpy as np
import pickle
import os

def word_features(wordlist, word_type, save, save_path):
    path = os.path.join(save_path, f'word_feature_{word_type}.pkl')
    if save==True:
        if word_type == 'fasttext':
            model = FastText.load_fasttext_format('fasttext/cc.zh.300.bin')
            vector = model.wv[wordlist] 
        elif word_type == 'bert':
            torch.cuda.set_device(8)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = BertTokenizer('bert/vocab.txt')
            model = BertModel.from_pretrained('bert/bert-base-chinese').to(device)
            model.eval()
            inputs = tokenizer(wordlist, return_tensors='pt')
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            vector = outputs.last_hidden_state[:, 1, :].cpu().numpy()
        print(vector.shape)

        word_vec = {}
        for idx in range(len(wordlist)):
            word_vec[wordlist[idx]] = vector[idx]
        with open(path, 'wb') as f:
            pickle.dump(word_vec, f)

    print('*********************loading***********************')
    with open(path, 'rb') as f:
        word_vec = pickle.load(f)


if __name__ == '__main__':
    wordlist = list(Label_Ref().label.keys())
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    word_type = 'fasttext'
    word_features(wordlist, word_type, save=True, save_path=save_path)
    word_type = 'bert'
    word_features(wordlist, word_type, save=True, save_path=save_path)
