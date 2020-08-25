import pandas as pd
import torch
from transformers import BertTokenizer

from load_data import load_data


def main():
    data_df = pd.read_csv('./datasets/Womens Clothing E-Commerce Reviews.csv')
    text_cols = ['Title', 'Review Text']
    cat_cols = ['Clothing ID', 'Division Name']
    num_cols = ['Rating']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = load_data(data_df, text_cols, tokenizer, label_col='Recommended IND',
                        categorical_cols=cat_cols, numerical_cols=num_cols,
                        sep_text_token_str=tokenizer.sep_token)
    print('here')

if __name__ == '__main__':
    main()