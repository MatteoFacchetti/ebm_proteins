# ebm_proteins

## Introduction

During their life, proteins are subject to mutations that affect their amino acid sequences. Some of these changes can be beneficial, other mutations can be dangerous.

The interest in modeling genetic mutations in order to determine their effects has been increasing among bioengineering and human health researchers, since many human diseases are often due to these amino-acid substitutions. Being able to successfully model such mutations can lead to correctly predict the effect of a genetic change on the human body and, ultimately, save lives.

The goal of this work is to build an unsupervised method to model the behavior of the proteins belonging to the beta-lactamase TEM-1 family. Ultimately, we will use this model to predict the effect of a change in the amino acid sequence on the resistance of the proteins against beta-lactam antibiotics.

This repository contains the code that I developed, with the help of my supervisor, professor Christoph Feinauer, and that lies at the heart of the Masters thesis I completed when I graduated in 2020. The full PDF document is available [here](https://github.com/MatteoFacchetti/ebm_proteins/blob/main/docs/Prediction%20of%20the%20Pathogenicity%20of%20Genetic%20Mutations%20with%20Energy-based%20Models.pdf).

## Main Results

This work proves that data science, machine and deep learning can play a crucial role to model protein sequences, and that significant results can be achieved even with a relatively simple model. In fact, with our model we are able to achieve a rank correlation of -0.26 between the MIC scores calculated in laboratory and the energy difference provided by our model between a real sequence and a mutated one. More info on this in the document available in this repository.

This is a neverending work in progress:  more computational power and more availability of resources will allow to build more advanced deep learning algorithms. Complex models will presumably improve the results even more, and for this reason the reader is encouraged to get in touch with me in case he or she has any interesting idea to improve this work.
