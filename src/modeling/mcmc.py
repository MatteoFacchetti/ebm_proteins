import numpy as np
import click
from copy import copy
import math
from tqdm import tqdm
import logging
import torch

from utils import data_utils as du
from modeling.mlp import encoding


@click.command()
@click.option('-i', '--n_iterations', default=6000000, type=int)
@click.option('-s', '--step_size', default=10000, type=int)
@click.option('-r', '--resume', default=True, type=bool)
def main(n_iterations=6000000, step_size=10000, resume=True):
    """
    Generate protein sequences using Markov-Chain Monte-Carlo methods.

    Parameters
    ----------
    n_iterations : int, default 6_000_000
        Number of total iterations before stopping the MCMC.
    step_size : int, default 10_000
        Save the generated sequence after `step_size` iterations.
    resume : bool, default True
        If True, resume a previously stopped MCMC, else start a new one.
    """
    logger.info(f"Running {n_iterations}, saving a sequence every {step_size} iterations.")
    logger.info(f"This process will generate {n_iterations // step_size} sequences.")
    # Load configuration file
    cfg = du.read_yaml("../config/config.yaml")
    parsed_sequences = cfg["parsed_sequence"]
    output = cfg["generated_sequence"]
    amino_acids = cfg["amino_acids"]
    parsed_sequences = du.read_yaml(parsed_sequences)

    if resume:
        # Begin from the last saved sequence
        logger.info(f"Resuming Markov Chain saved at {output}")
        try:
            old_seq = du.read_yaml(output)[-1]
        except (TypeError, FileNotFoundError):
            logger.info(f"Saved chain not found. Starting a new one. Output file: {output}")
            old_seq = parsed_sequences[np.random.randint(0, len(parsed_sequences))]
    else:
        # Start from a random sequence
        logger.info(f"Starting a new Markov Chain. Output file: {output}")
        old_seq = parsed_sequences[np.random.randint(0, len(parsed_sequences))]

    # Begin iterations
    logger.info(f"Running {n_iterations} iterations")

    # Build neural network
    input_size, hidden_size, output_size = 330 * 21, 100, 1

    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size)
    )

    encoded_old = encoding(old_seq)
    old_E = model(encoded_old)

    for step in tqdm(range(n_iterations)):
        # Sample a random position `i` and a random amino acid `a`
        aa_subset = copy(amino_acids)
        i = np.random.randint(0, len(old_seq))
        aa_subset.remove((old_seq[i]))
        a = np.random.choice(aa_subset)

        # Generate new sequence
        new_seq = copy(old_seq)
        new_seq = list(new_seq)
        new_seq[i] = a
        new_seq = "".join(new_seq)

        # Calculate energy of old and new sequences
        encoded_new = encoding(new_seq)
        new_E = model(encoded_new)

        # Accept with probability exp{-(energy_difference)}
        p = math.exp(old_E - new_E)
        u = np.random.uniform()

        if u < p:
            old_seq, old_E, encoded_old = copy(new_seq), copy(new_E), copy(encoded_new)
        if step % step_size == 0:
            # Saving one sequence every 10_000 iterations
            du.append_yaml(old_seq, output)


def sample_sequences(model, old_seq, amino_acids, n_iterations, step_size):
    """
    Sample protein sequences using Markov-Chain Monte-Carlo methods starting from a given sequence and a given model.

    Parameters
    ----------
    model : torch.nn.Sequential
        Neural network used to calculate the energy of the sequence.
    old_seq : str
        Initial sequence that the algorithm takes to begin the sampling.
    amino_acids : list
        List of amino-acids.
    n_iterations : int
        Total number of iterations.
    step_size : int
        Save one sequence every `step_size` iterations.

    Returns
    -------
    sampled_sequences : list
        List containing `n_iterations`  / `step_size` sampled sequences.
    """

    encoded_old = encoding(old_seq)
    old_E = model(encoded_old)

    sampled_sequences = []

    for step in tqdm(range(n_iterations)):
        # Sample a random position `i` and a random amino acid `a`
        aa_subset = copy(amino_acids)
        i = np.random.randint(0, len(old_seq))
        aa_subset.remove((old_seq[i]))
        a = np.random.choice(aa_subset)

        # Generate new sequence
        new_seq = copy(old_seq)
        new_seq = list(new_seq)
        new_seq[i] = a
        new_seq = "".join(new_seq)

        # Calculate energy of old and new sequences
        encoded_new = encoding(new_seq)
        new_E = model(encoded_new)

        # Accept with probability exp{-(energy_difference)}
        p = math.exp(old_E - new_E)
        u = np.random.uniform()

        if u < p:
            old_seq, old_E, encoded_old = copy(new_seq), copy(new_E), copy(encoded_new)
        if step % step_size == 0 and step != 0:
            # Saving one sequence every 10_000 iterations
            sampled_sequences.append(old_seq)

    return sampled_sequences


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
