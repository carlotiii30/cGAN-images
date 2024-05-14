import os
import re
import numpy as np
from textblob import TextBlob
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


class Text:
    """Class that provides functions for processing text.

    This class contains methods for performing preprocessing and
    text representation operations.

    """

    @staticmethod
    def process_text(text):
        """Processes a given text.

        Performs preprocessing, tokenization, and representation
        operations on the text.

        Args:
            text (str): The text to process.

        Returns:
            str: The preprocessed text.
            list: List of tokens obtained from the text.
            np.array: Numerical representation of the text using GloVe.
        """
        # Text preprocessing
        preprocessed_text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
        preprocessed_text = preprocessed_text.lower()  # Convert to lowercase
        preprocessed_text = preprocessed_text.rstrip()  # Remove trailing spaces

        # Tokenization
        tokens = TextBlob(preprocessed_text).words

        # Convert GloVe vectors to Word2Vec format
        glove_input_twitter = "./data/glove/glove.twitter.27B.50d.txt"
        word2vec_output_twitter = "./data/word2vec/glove_model1.word2vec"
        if os.path.exists(glove_input_twitter) and not os.path.exists(
            word2vec_output_twitter
        ):
            glove2word2vec(glove_input_twitter, word2vec_output_twitter)

        glove_input_wiki = "./data/glove/glove.6B.100d.txt"
        word2vec_output_wiki = "./data/word2vec/glove_model2.word2vec"
        if os.path.exists(glove_input_wiki) and not os.path.exists(
            word2vec_output_wiki
        ):
            glove2word2vec(glove_input_wiki, word2vec_output_wiki)

        # Load the pre-trained GloVe models
        model1 = KeyedVectors.load_word2vec_format(
            glove_input_twitter, binary=False, no_header=True
        )
        model2 = KeyedVectors.load_word2vec_format(
            glove_input_wiki, binary=False, no_header=True
        )

        # Text representation (GloVe)
        numerical_representation1 = np.zeros((len(tokens), model1.vector_size))
        numerical_representation2 = np.zeros((len(tokens), model2.vector_size))
        for i, token in enumerate(tokens):
            if token in model1:
                numerical_representation1[i] = model1[token]
            if token in model2:
                numerical_representation2[i] = model2[token]

        return (
            preprocessed_text,
            tokens,
            numerical_representation1,
            numerical_representation2,
        )
