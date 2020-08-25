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

from model.multimodal_config import TabularConfig
from model.multimodal_transformers import (
    BertWithTabular, RobertaWithTabular, DistilBertWithTabular
)
from model.multimodal_modeling_auto import AutoModelWithTabular

MODEL_CLASSES = {
    'bert_w_tabular': (BertConfig, BertWithTabular, BertTokenizer, 'bert-base-uncased'),
    'roberta_w_tabular': (RobertaConfig, RobertaWithTabular, RobertaTokenizer, 'roberta-base'),
    'finbert_w_tabular': (BertConfig, AutoModelWithTabular, AutoTokenizer, 'ipuneetrathore/bert-base-cased-finetuned-finBERT'),
    'distilbert_w_tabular': (DistilBertConfig, DistilBertWithTabular, DistilBertTokenizer, 'distilbert-base-uncased')
}


model = 'distilbert_w_tabular'  # change this to try different with tabular models

# Load pre-trained model tokenizer (vocabulary)
tokenizer = MODEL_CLASSES[model][2].from_pretrained(MODEL_CLASSES[model][3])

text1 = ['Who was Jim Henson ?', 'Jim Henson was a puppeteer']
text1 = f' {tokenizer.sep_token} '.join(text1)
text2 = 'Just some random text'

print(f'======= Texts ======\n{text1}\n{text2}\n')

# Tokenize input
model_input = tokenizer([text1, text2], padding=True, truncation=True, return_tensors='pt')
tokenized_str1 = ' '.join(tokenizer.convert_ids_to_tokens(model_input['input_ids'][0]))
tokenized_str2 = ' '.join(tokenizer.convert_ids_to_tokens(model_input['input_ids'][0]))
print(f'====== Tokenized Texts ======:\n{tokenized_str1}\n{tokenized_str2}\n')
print(f"Input ids shape: {model_input.data['input_ids'].shape}")

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


model_input = model_input.to(device)
numerical_feat = numerical_feat.to(device)
categorical_feat = categorical_feat.to(device)
labels = labels.to(device)
model.to(device)


optimizer = AdamW(model.parameters(), lr=5e-5)
# One forward pass and one backward pass
model.train()
loss, _, _ = model(**model_input,
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
    _, logits, classifier_outputs = model(**model_input,
                                          cat_feats=categorical_feat,
                                          numerical_feats=numerical_feat
                                          )
    print(f'Logits:\n{logits.detach().numpy()}')  # shape: batch_size x num_labels
    # The classifier_output is specified by last_ith_layer_as_embd of the bert config
    # The input, hidden, and output embeddings of the final classifier are saved as a list of tensors
    # last_ith_layer_as_embd indexes into this
    print(f'Classifier outputs shape: {classifier_outputs[-2].shape}')