from os.path import dirname, abspath, join
import sys

import torch
from transformers import BertConfig, BertTokenizer, AdamW

# make sure can import modules from the current directory
cur_folder = dirname(abspath(''))
src_folder = join(cur_folder, 'src')
sys.path.insert(0, cur_folder)
sys.path.insert(1, src_folder)

from multimodal_config import TabularConfig
from multimodal_transformers import BertWithTabular

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
text1 = '[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]'
text2 = '[CLS] Just some random text [SEP]'
tokenized_text1 = tokenizer.tokenize(text1)
tokenized_text2 = tokenizer.tokenize(text2)

# Convert token to vocabulary indices
indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
#  Need to do padding for the shorter sentence
indexed_tokens2 = indexed_tokens2 + [0] * (len(indexed_tokens1) - len(indexed_tokens2))
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids1 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
segments_ids2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens1, indexed_tokens2])
segments_tensors = torch.tensor([segments_ids1, segments_ids2])

# 5 numerical features
numerical_feat = torch.rand(2, 5).float()
print(numerical_feat.shape)
# 9 categorical features
categorical_feat = torch.tensor([[0, 0, 0, 1, 0, 1, 0, 1, 0],
                                 [1, 0, 0, 0, 1, 0, 1, 0, 0]]).float()
labels = torch.tensor([1, 0])


# See BasicBertConfig for more details on configuring Bert
tabular_config = TabularConfig(
    combine_feat_method='gating_on_cat_and_num_feats_then_sum',  # change this to specify the method of combining tabular data
    cat_feat_dim=9,  # need to specify this
    numerical_feat_dim=5,  # need to specify this
    num_labels=2,   # need to specify this
    use_num_bn=False,  # No bn since we only have one training example for demonstration purposes
)

bert_config = BertConfig()
bert_config.tabular_config = tabular_config

device = str('cuda' if torch.cuda.is_available() else 'cpu')

# loads the pre-trained weights for Bert module
model = BertWithTabular.from_pretrained('bert-base-uncased', config=bert_config)
print(model)


tokens_tensors = tokens_tensor.to(device)
segments_tensors = segments_tensors.to(device)
numerical_feat = numerical_feat.to(device)
categorical_feat = categorical_feat.to(device)
labels = labels.to(device)
model.to(device)


optimizer = AdamW(model.parameters(), lr=5e-5)
# One forward pass and one backward pass
model.train()
loss, _, _ = model(tokens_tensors, token_type_ids=segments_tensors,
                   labels=labels, cat_feats=categorical_feat,
                   numerical_feats=numerical_feat)
print(f'Loss: {loss.item()}')
loss.backward()
optimizer.step()
model.zero_grad()

# Inference and/or evaluation
model.eval()
with torch.no_grad():
    # If labels are passed then loss will also be returned
    _, logits, classifier_outputs = model(tokens_tensors,
                                          token_type_ids=segments_tensors,
                                          cat_feats=categorical_feat,
                                          numerical_feats=numerical_feat
                                          )
    print(f'Logits:\n{logits.detach().numpy()}')  # shape: batch_size x num_labels
    # The classifier_output is specified by last_ith_layer_as_embd of the bert config
    # The input, hidden, and output embeddings of the final classifier are saved as a list of tensors
    # last_ith_layer_as_embd indexes into this
    print(f'Classifier outputs shape: {classifier_outputs[-2].shape}')