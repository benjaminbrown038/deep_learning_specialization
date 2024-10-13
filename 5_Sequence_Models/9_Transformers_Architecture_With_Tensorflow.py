import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from public_tests import *

from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForTokenClassification

def get_angles():

    return angles

def positional_encoding():

    return tf.cast(pos_encoding,dtype = tf.float32)


def create_padding_mask(decoder_token_ids):

    return seq[:, tf.newaxis, :]


def create_look_ahead_mask(sequence_length):

    return mask


def scaled_dot_product_attention():

    return output, attention_weights


def FullyConnected(embedding_dim, fully_connected_dim):

    return 


class EncoderLayer():

      def __init__():


      def call():



EncoderLayer_test(EncoderLayer)


class Encoder():

      def __init__():

      def call():
  


Encoder_test(Encoder)



class DecoderLayer(tf.keras.layers.Layer):

      def __init__():


      def call():


DecoderLayer_test(DecoderLayer, create_look_ahead_mask)


class Decoder():

      def __init__():

      def call():


Decoder_test(Decoder, create_look_ahead_mask, create_padding_mask)



class Transformer():

      def __init__():


      def call():



  Transformer_test(Transformer, create_look_ahead_mask, create_padding_mask)

