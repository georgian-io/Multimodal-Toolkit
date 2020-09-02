# Combine Methods

This page explains the methods that are supported by `multimodal.tabular_combiner.TabularFeatCombiner`.
See the table below and [here](https://docs.google.com/document/d/1rmsUeg1QnWed-M8VwD9Vwt6FvlAsarnS7eWWnD8nH4Y/edit) for details

If you have rich categorical and numerical features any of the attention, gating, weighted sum methods are worth trying. 

| Combine Feat Method | Description | requires both cat and num features | 
|:--------------|:-------------------|:-------|
| text_only | Uses just the text columns as processed by Bert before final classifier layer(s). Essentially equivalent to HuggingFace's `BertForSequenceClassification` |  False | 
| concat | Concatenate Bert output, numerical feats, and categorical feats all at once before final classifier layer(s) | False |
| mlp_on_categorical_then_concat | MLP on categorical feats then concat bert output, numerical feats, and processed categorical feats before final classifier layer(s) | False (Requires cat feats)
| individual_mlps_on_cat_and_numerical_feats_then_concat | Separate MLPs on categorical feats and numerical feats then concatenation of Bert output, with processed numerical feats, and processed categorical feats before final classifier layer(s). | False
| mlp_on_concatenated_cat_and_numerical_feats_then_concat | MLP on concatenated categorical and numerical feat then concatenated with Bert output before final classifier layer(s) | True
| attention_on_cat_and_numerical_feats | Attention based summation of Bert outputs, numerical feats, and categorical feats queried by bert outputs before final classifier layer(s). | False
| gating_on_cat_and_num_feats_then_sum | Gated summation of bert outputs, numerical feats, and categorical feats before final classifier layer(s). Inspired by [Integrating Multimodal Information in Large Pretrained Transformers](https://www.aclweb.org/anthology/2020.acl-main.214.pdf) which performs the mechanism for each token. | False
| weighted_feature_sum_on_bert_cat_and_numerical_feats | Learnable weighted feature-wise sum of Bert outputs, numerical feats and categorical feats for each feature dimension before final classifier layer(s) | False

