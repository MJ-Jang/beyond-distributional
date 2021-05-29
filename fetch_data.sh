mkdir resources

# 1. LAMA
# Download and unzip
wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip

# Move ConceptNet data to resources folder
mv data/ConceptNet/test.jsonl resources/

# Remove unncessary files
rm -rf data.zip
rm -rf data

# 2. SNLI
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip


# Move ConceptNet data to resources folder
mv snli_1.0/snli_1.0_dev.jsonl resources/
mv snli_1.0/snli_1.0_test.jsonl resources/
mv snli_1.0/snli_1.0_train.jsonl resources/

# Remove unncessary files
rm -rf snli_1.0.zip
rm -rf snli_1.0