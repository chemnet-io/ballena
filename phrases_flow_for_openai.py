# %%

import pandas as pd
import nltk
import re


def phrases_slicing(x):
    phrases = []
    slices = round(len(x) / 2048)
    for i in range(slices):
        if i + 1 == slices:
            phrases.append(x[i * 2048:])
        else:
            phrases.append(x[i * 2048:(i + 1) * 2048])
    return phrases


def remove_not_readable(text):  # "\"#$%&\'()*+/:;<=>?@[\\]^_`{|}~\n\n\t`˚-–<’>0123456789‘©°−"
    return text.translate(str.maketrans('', '', "�"))


def remove_split(text):
    return re.sub('- ', '', text)


def phrases_flow():
    path = 'data/'

    texts_df = pd.read_parquet('{}texts_query17mai18-18.parquet'.format(path))

    text_column = 'text'

    texts_df[text_column] = texts_df[text_column].apply(lambda x: x.replace('\n', ''))
    texts_df[text_column] = texts_df[text_column].apply(lambda x: x.replace('  ', ''))
    texts_df[text_column] = texts_df[text_column].apply(lambda x: x.replace('- ', ''))

    texts_df[text_column] = texts_df[text_column].apply(lambda x: remove_split(x))
    texts_df[text_column] = texts_df[text_column].apply(lambda x: remove_not_readable(x))
    # text_df add column phrases
    texts_df['phrases'] = texts_df[text_column].apply(lambda x: phrases_slicing(x))

    print(texts_df)

    texts_df.to_parquet("{}pdf-phrases17mai18-23.parquet".format(path))
    #print(texts_df.head())


if __name__ == "__main__":
    phrases_flow()
