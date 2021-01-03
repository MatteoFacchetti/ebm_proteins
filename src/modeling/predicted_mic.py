import pandas as pd
from copy import copy
from tqdm import tqdm

import torch

from utils import data_utils as du
from modeling.mlp import encoding

mic = pd.read_excel("data/sd01.xlsx")
mic = mic[["Residue", "Wt_AA", "Mutant_AA", "MIC_Score", "∆∆G_foldX"]]
mic["predicted_MIC"] = 0

cfg = du.read_yaml("../config/config.yaml")

pos = []
for sequence in du.read_yaml(cfg["parsed_sequence"]):
    if "X" not in sequence:
        pos.append(sequence)

seqs = [seq for seq in pos if seq[24] == "H"]

model = torch.load(cfg["output_model"])

predictions = pd.DataFrame()

for i in tqdm(range(len(mic))):
    pred = pd.DataFrame()
    mic_value = mic.loc[i, "MIC_Score"]
    delta_g = mic.loc[i, "∆∆G_foldX"]
    residue = mic.loc[i, "Residue"]
    wt_aa = mic.loc[i, "Wt_AA"]
    mutant_aa = mic.loc[i, "Mutant_AA"]
    seqs = [seq for seq in pos if seq[residue] == wt_aa]

    for seq in seqs:
        # Mutate sequence
        new_seq = copy(seq)
        new_seq = list(new_seq)
        new_seq[residue] = mutant_aa
        new_seq = "".join(new_seq)

        # Calculate energy of original and mutated sequences
        encoded_new = encoding(new_seq)
        new_E = model(encoded_new)
        old_E = model(encoding(seq))

        predicted_mic = float(new_E) - float(old_E)

        prediction = pd.DataFrame({
            "Residue": residue,
            "Wt_AA": wt_aa,
            "Mutant_AA": mutant_aa,
            "MIC_Score": mic_value,
            "∆∆G_foldX": delta_g,
            "∆E": predicted_mic
        }, index=[seq])

        pred = pred.append(prediction)

    predicted_MIC = pred["∆E"].mean()
    prediction = pd.DataFrame({
        "Residue": residue,
        "Wt_AA": wt_aa,
        "Mutant_AA": mutant_aa,
        "MIC_Score": mic_value,
        "∆∆G_foldX": delta_g,
        "∆E": predicted_MIC
    }, index=[i])

    predictions = predictions.append(prediction)

predictions.to_csv("data/predictions.csv")
