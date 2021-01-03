import click
import logging
from tqdm import tqdm
from copy import deepcopy
import torch
import random
import numpy as np

from utils import data_utils as du
from modeling.mlp import encoding
from modeling.mcmc import sample_sequences


@click.command()
@click.option('-e', '--n_epochs', type=int)
@click.option('-l', '--learning_rate', default=0.001, type=float)
def main(n_epochs, learning_rate=0.001):
    logger.info(f"epochs: {n_epochs}")
    logger.info(f"learning rate: {learning_rate}")
    # Load sequences
    cfg = du.read_yaml("../config/config.yaml")
    output_path = cfg["output_model"]

    parsed_sequences = cfg["parsed_sequence"]
    amino_acids = cfg["amino_acids"]
    parsed_sequences = du.read_yaml(parsed_sequences)
    old_seq = parsed_sequences[np.random.randint(0, len(parsed_sequences))]

    logger.info(f"The model will be saved at {output_path}")
    logger.info("Loading positive samples")
    poss = []

    for sequence in du.read_yaml(cfg["parsed_sequence"]):
        if "X" not in sequence:
            poss.append(sequence)

    # Initialize MLP
    input_size, hidden_size, output_size = 330 * 21, 100, 1

    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size)
    )

    # Begin training
    for epoch in range(n_epochs):
        logger.info(f"Epoch n. {epoch + 1} out of {n_epochs}")
        logger.info("Sampling sequences")
        neg = sample_sequences(model, old_seq, amino_acids, n_iterations=3_000_000, step_size=3_000)

        # The next sampling will start from a new real sequence (contrastive divergence)
        old_seq = parsed_sequences[np.random.randint(0, len(parsed_sequences))]

        # Compute gradient of the parameters of the network
        logger.info("Computing gradient and updating the model")
        neg_grads = []  # Gradient for the negative samples
        for seq in neg:
            encoded = encoding(seq)  # Encode the sequence
            E = model(encoded)  # Calculate energy
            model.zero_grad()
            E.backward()  # Compute gradient

            new_grads = [deepcopy(param.grad) for param in model.parameters()]
            neg_grads.append(new_grads)  # Save gradient

        pos = random.sample(poss, k=1000)
        pos_grads = []  # Gradient for the positive samples
        for seq in pos:
            encoded = encoding(seq)
            E = model(encoded)
            model.zero_grad()
            E.backward()

            new_grads = [deepcopy(param.grad) for param in model.parameters()]
            pos_grads.append(new_grads)

        neg_0_weight, neg_0_bias, neg_2_weight, neg_2_bias = parameters_grads(neg_grads)
        pos_0_weight, pos_0_bias, pos_2_weight, pos_2_bias = parameters_grads(pos_grads)

        # Compute the expected value of each parameter
        exp_neg_0_weight, exp_neg_0_bias, exp_neg_2_weight, exp_neg_2_bias = compute_expectation(
            neg_0_weight, neg_0_bias, neg_2_weight, neg_2_bias
        )
        exp_pos_0_weight, exp_pos_0_bias, exp_pos_2_weight, exp_pos_2_bias = compute_expectation(
            pos_0_weight, pos_0_bias, pos_2_weight, pos_2_bias
        )

        # Calculate difference of expectation of the gradient of the parameters between positive and negative samples
        diff_0_weight = exp_pos_0_weight - exp_neg_0_weight
        diff_0_bias = exp_pos_0_bias - exp_neg_0_bias
        diff_2_weight = exp_pos_2_weight - exp_neg_2_weight
        diff_2_bias = exp_pos_2_bias - exp_neg_2_bias

        loss_grad = [diff_0_weight, diff_0_bias, diff_2_weight, diff_2_bias]

        # Update the parameters
        i = 0
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * loss_grad[i]
                i += 1

        # Free memory before next epoch
        del neg_grads
        del pos_grads
        del neg_0_weight
        del neg_0_bias
        del neg_2_weight
        del neg_2_bias
        del pos_0_weight
        del pos_0_bias
        del pos_2_weight
        del pos_2_bias

    # Save the trained model
    logger.info("Saving model")
    torch.save(model, output_path)
    logger.info("Done")


def intersection(lst1, lst2):
    """
    Intersection of two lists.
    """
    return list(set(lst1) & set(lst2))


def parameters_grads(grads):
    """
    Split a list containing the gradient of the parameters into a separate list of gradients for each parameter.

    Parameters
    ----------
    grads : list
    """
    weight_0 = [grads[i][0] for i in range(len(grads))]
    bias_0 = [grads[i][1] for i in range(len(grads))]
    weight_2 = [grads[i][2] for i in range(len(grads))]
    bias_2 = [grads[i][3] for i in range(len(grads))]

    return weight_0, bias_0, weight_2, bias_2


def compute_expectation(weight_0, bias_0, weight_2, bias_2):
    """
    Compute the expected value of the gradient of each parameter.
    """
    expectation_0_weight = sum(weight_0) / len(weight_0)
    expectation_0_bias = sum(bias_0) / len(bias_0)
    expectation_2_weight = sum(weight_2) / len(weight_2)
    expectation_2_bias = sum(bias_2) / len(bias_2)

    return expectation_0_weight, expectation_0_bias, expectation_2_weight, expectation_2_bias


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
