# Sami dialects

The repository contains analysis and detection methods for the analysis and automatic detection of Sami dialects. This involves the analysis of the differences in speech of Sami dialects and the automatic detection of the different dialects. Our aim is to increase our understanding of the prosodic differences of these dialects and use them in evaluating them.

# Setup
Instructions for setting up the environment using conda will be added later.

# Running the code
The code includes options for extracting a range of acoustic features and functionals as well as few different classification methods. In addition, the code supports visualization methods targeting interpretability of the results in terms of the used features.

```console
# Extract mfccs and classify data with a random forest classifier
python3 main.py \
    --no-preprocess_all \
    --no-explain \
    --datafile /path/to/datafile.csv \
    --datadir /path/to/datafolder \
    --target_label label_name \
    --feature_set kaldi \
    --feature_subset mfccs \
    --pooling_method meanstd \
    --classifier rforest \
    --feature_file None \
    --portion 1.0 \
    --seed 123456 \
    --split split_small
```
