from utils import data_utils as du

cfg = du.read_yaml("../config/config.yaml")
raw_sequence = cfg["raw_sequence"]
output = cfg["parsed_sequence"]

fasta_sequences = du.read_sequences(raw_sequence)

sequences = du.parse_fasta_sequences(fasta_sequences)

du.save_yaml(sequences, output)
