from os.path import dirname, abspath, join
import sys

import torch
from transformers.configuration_auto import (
    BertConfig, RobertaConfig, DistilBertConfig,
)
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    AutoTokenizer,
    DistilBertTokenizer,
    AdamW
)

# make sure can import modules from the current directory
cur_folder = dirname(abspath(''))
src_folder = join(cur_folder, 'src')
sys.path.insert(0, cur_folder)
sys.path.insert(1, src_folder)

from multimodal_config import TabularConfig
from multimodal_transformers import (
    BertWithTabular, RobertaWithTabular, DistilBertWithTabular
)
from multimodal_modeling_auto import AutoModelWithTabular

MODEL_CLASSES = {
    'bert_w_tabular': (BertConfig, BertWithTabular, BertTokenizer, 'bert-base-uncased'),
    'roberta_w_tabular': (RobertaConfig, RobertaWithTabular, RobertaTokenizer, 'roberta-base'),
    'finbert_w_tabular': (BertConfig, AutoModelWithTabular, AutoTokenizer, 'ipuneetrathore/bert-base-cased-finetuned-finBERT'),
    'distilbert_w_tabular': (DistilBertConfig, DistilBertWithTabular, DistilBertTokenizer, 'distilbert-base-uncased')
}


model = 'distilbert_w_tabular'  # change this to try different with tabular models

# Load pre-trained model tokenizer (vocabulary)
tokenizer = MODEL_CLASSES[model][2].from_pretrained(MODEL_CLASSES[model][3])

# Tokenize input
if model in ['bert_w_tabular', 'finbert_w_tabular', 'distilbert_w_tabular']:
    text1 = 'Who was Jim Henson ?', 'Jim Henson was a puppeteer'
    text2 = 'Just some random text'

    tokenized_text1 = tokenizer.tokenize(text1[0]), tokenizer.tokenize(text1[1])
    tokenized_text2 = tokenizer.tokenize(text2)
    # Convert token to vocabulary indices
    indexed_tokens1 = (tokenizer.convert_tokens_to_ids(tokenized_text1[0]),
                       tokenizer.convert_tokens_to_ids(tokenized_text1[1]))
    indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)

    # Add special tokens
    tokenized_ids1 = tokenizer.build_inputs_with_special_tokens(indexed_tokens1[0],
                                                                indexed_tokens1[1])
    tokenized_ids2 = tokenizer.build_inputs_with_special_tokens(indexed_tokens2)

    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids1 = tokenizer.create_token_type_ids_from_sequences(*indexed_tokens1)
    segments_ids2 = tokenizer.create_token_type_ids_from_sequences(indexed_tokens2)

    #  Need to do padding for the shorter sentence
    tokenized_ids2 = tokenized_ids2 + [0] * (len(tokenized_ids1) - len(tokenized_ids2))
    segments_ids2 = segments_ids2 + [0] * (len(segments_ids1) - len(segments_ids2))

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([tokenized_ids1, tokenized_ids2])
    if model in ['bert_w_tabular', 'finbert_w_tabular']:
        segments_tensors = torch.tensor([segments_ids1, segments_ids2])
    else:
        segments_tensors = None

elif model == 'roberta_w_tabular':
    text1 = 'Who was Jim Hensen ? </s> Jim Henson was a puppeteer'
    text2 = 'Just some random text'
    indexed_tokens1 = tokenizer(text1)['input_ids']
    indexed_tokens2 = tokenizer(text2)['input_ids']
    indexed_tokens2 = indexed_tokens2 + [0] * (len(indexed_tokens1) - len(indexed_tokens2))
    tokens_tensor = torch.tensor([indexed_tokens1, indexed_tokens2])
    segments_tensors = None

print(f'input ids shape: {tokens_tensor.shape}')

# 5 numerical features
numerical_feat = torch.rand(2, 5).float()
# 9 categorical features
categorical_feat = torch.tensor([[0, 0, 0, 1, 0, 1, 0, 1, 0],
                                 [1, 0, 0, 0, 1, 0, 1, 0, 0]]).float()
labels = torch.tensor([1, 0])


# See BasicBertConfig for more details on configuring Bert
tabular_config = TabularConfig(
    combine_feat_method='attention_on_cat_and_numerical_feats',  # change this to specify the method of combining tabular data
    cat_feat_dim=9,  # need to specify this
    numerical_feat_dim=5,  # need to specify this
    num_labels=2,   # need to specify this
    use_num_bn=False,  # No bn since we only have one training example for demonstration purposes
)

hf_config = MODEL_CLASSES[model][0].from_pretrained(MODEL_CLASSES[model][3])
hf_config.tabular_config = tabular_config


device = str('cuda' if torch.cuda.is_available() else 'cpu')

# loads the pre-trained weights for Bert module
model = MODEL_CLASSES[model][1].from_pretrained(MODEL_CLASSES[model][3], config=hf_config)
print(model)


tokens_tensors = tokens_tensor.to(device)
segments_tensors = segments_tensors.to(device) if segments_tensors is not None else None
numerical_feat = numerical_feat.to(device)
categorical_feat = categorical_feat.to(device)
labels = labels.to(device)
model.to(device)


optimizer = AdamW(model.parameters(), lr=5e-5)
# One forward pass and one backward pass
model.train()
if segments_tensors is not None:
    loss, _, _ = model(tokens_tensors, token_type_ids=segments_tensors,
                       labels=labels, cat_feats=categorical_feat,
                       numerical_feats=numerical_feat)
else:
    loss, _, _ = model(tokens_tensors,
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
    if segments_tensors is not None:
        _, logits, classifier_outputs = model(tokens_tensors,
                                              token_type_ids=segments_tensors,
                                              cat_feats=categorical_feat,
                                              numerical_feats=numerical_feat
                                              )
    else:
        _, logits, classifier_outputs = model(tokens_tensors,
                                              cat_feats=categorical_feat,
                                              numerical_feats=numerical_feat
                                              )
    print(f'Logits:\n{logits.detach().numpy()}')  # shape: batch_size x num_labels
    # The classifier_output is specified by last_ith_layer_as_embd of the bert config
    # The input, hidden, and output embeddings of the final classifier are saved as a list of tensors
    # last_ith_layer_as_embd indexes into this
    print(f'Classifier outputs shape: {classifier_outputs[-2].shape}')