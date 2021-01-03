import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import re

from Bio import SeqIO


def read_yaml(yaml_file):
    """
    Read a yaml file.

    Parameters
    ----------
    yaml_file : str
        Path to file.
    """
    with open(yaml_file, 'r') as ymlfile:
        yml = yaml.load(ymlfile, yaml.SafeLoader)
    return yml


def save_yaml(output, path):
    """
    Save a yaml file.

    Parameters
    ----------
    output
        Object we want to save as yaml file.
    path : str
        Path to output file.
    """
    with open(path, 'w') as outfile:
        yaml.dump(output, outfile, default_flow_style=False)


def append_yaml(output, path):
    """
    Append an object to an existing yaml file.

    Parameters
    ----------
    output
        Object we want to append to the yaml file.
    path : str
        Path to yaml file.
    """
    with open(path, 'a') as outfile:
        yaml.dump([output], outfile, default_flow_style=False)


def read_sequences(path, format='fasta'):
    """
    Read a list of protein sequences of the specified format.

    Parameters
    ----------
    path : str
        Path to the file containing the list of sequences.
    format : str, default 'fasta'
        Format of the proteine sequences.

    Returns
    -------
    A list containing all the sequences.
    """
    return SeqIO.parse(open(path), format)


def remove_lower(string):
    """
    Remove all the lowercase characters contained in a string.

    Parameters
    ----------
    string : str

    Returns
    -------
    The input string without all the lowercase letters.
    """
    return re.sub('[a-z]', '', string)


def parse_fasta_sequences(fasta_sequences):
    """
    Clean protein sequences (in FASTA format) by removing dots and lowercase letters.

    Parameters
    ----------
    fasta_sequences : list
        List of protein sequences in FASTA format.

    Returns
    -------
    sequences : list
        List of strings, each containing a protein sequence.
    """
    sequences = []

    for fasta in fasta_sequences:
        sequence = str(fasta.seq)
        sequence = remove_lower(sequence.replace(".", ""))
        sequences.append(sequence)

    return sequences


def _one_hot_encode(seq, show_matrix=False):
    codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

    missing = list(set(codes) - set(seq))
    df = pd.DataFrame(list(seq))
    missing_df = pd.DataFrame(np.zeros((len(seq), len(missing)), dtype=int), columns=missing)
    encoded_matrix = df[0].str.get_dummies(sep=',')
    encoded_matrix = encoded_matrix.join(missing_df)
    encoded_matrix = encoded_matrix.sort_index(axis=1)
    flattened = encoded_matrix.values.flatten()

    if show_matrix:
        cmap = sns.light_palette("seagreen", as_cmap=True)
        display(encoded_matrix.style.background_gradient(cmap=cmap))

    return flattened


def one_hot_encode(seq):
    """
    Encode a protein sequence into a one-dimensional numpy array.

    Parameters
    ----------
    seq: str
        Protein sequence.

    Returns
    -------
    encoded : numpy.array
    """
    codes = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    length_codes = len(codes)
    length_sequence = len(seq)
    encoded = np.zeros(length_codes * length_sequence)
    for i, char in enumerate(seq):
        encoded[i * length_codes + codes.index(char)] = 1
    return encoded
