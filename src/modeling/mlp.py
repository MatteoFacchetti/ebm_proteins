import torch
import logging

from utils import data_utils as du

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_fmt)
logger = logging.getLogger(__name__)


def encoding(sequence):
    """
    Encode a proteine sequence into a one-dimensional tensor.

    Parameters
    ----------
    sequence : str
        Protein sequence.

    Returns
    -------
    encoded_sequence : torch.Tensor

    Examples
    --------
    >>> encoding("-AVQN")
    tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    """
    encoded_sequence = du.one_hot_encode(sequence)
    encoded_sequence = torch.Tensor(encoded_sequence)
    return encoded_sequence
