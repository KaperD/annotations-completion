import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer


def encode_names(column, train_size, column_name, max_new_columns):
    """
    Converts column of camelCase names to 100 columns with most popular words
    with 1 (if name contains word) and 0 (otherwise)
    """

    def split_camel_case(x):
        words = [word.lower() for word in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', x)]
        return ' '.join(words)

    vectorizer = CountVectorizer(preprocessor=split_camel_case, max_features=max_new_columns)
    vectorizer.fit(column[:train_size])
    new_columns = vectorizer.transform(column).toarray()
    new_names = [column_name + '_' + name for name in vectorizer.get_feature_names_out()]
    return new_columns, new_names


def encode_lists(column, train_size, column_name, max_new_columns):
    """
    Converts column of lists of words to 100 columns with most popular words
    with 1 (if list contains word) and 0 (otherwise)
    """

    joined_words = np.array([' '.join(x).replace('.', '') for x in column])

    vectorizer = CountVectorizer(max_features=max_new_columns)
    vectorizer.fit(joined_words[:train_size])
    new_columns = vectorizer.transform(joined_words).toarray()
    new_names = [column_name + '_' + name for name in vectorizer.get_feature_names_out()]
    return new_columns, new_names


def encode_column(column, train_size, column_name, max_new_columns):
    """
    Converts column of some type to column of integers
    """

    if isinstance(column[0], str):
        if (column[:train_size] == '').all():
            return None, []
        return encode_names(column, train_size, column_name, max_new_columns)
    elif isinstance(column[0], list):
        if np.alltrue([x == [] for x in column[:train_size]]):
            return None, []
        return encode_lists(column, train_size, column_name, max_new_columns)
    else:
        return np.array([column.astype(np.int64)]).T, [column_name]
