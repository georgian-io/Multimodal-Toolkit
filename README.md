# Multimodal Transformers | Transformers with Tabular Data

--------------------------------------------------------------------------------
**[Documentation](https://multimodal-toolkit.readthedocs.io/en/latest/index.html)** | **[Colab Notebook](https://multimodal-toolkit.readthedocs.io/en/latest/notes/colab_example.html)** | **[Blog Post](https://medium.com/georgian-impact-blog/how-to-incorporate-tabular-data-with-huggingface-transformers-b70ac45fcfb4)**

A toolkit for incorporating multimodal data on top of text data for classification
and regression tasks. It uses HuggingFace transformers as the base model for text features.
The toolkit adds a combining module that takes the outputs of the transformer in addition to categorical and numerical features
to produce rich multimodal features for downstream classification/regression layers.
Given a pretrained transformer, the parameters of the combining module and transformer are trained based
on the supervised task. For a brief literature review, check out the accompanying [blog post](https://medium.com/georgian-impact-blog/how-to-incorporate-tabular-data-with-huggingface-transformers-b70ac45fcfb4) on Georgian's Impact Blog. 

![](https://drive.google.com/uc?export=view&id=1aMNrv5kHDcaq8gS1EFtA6Ri4Tg_aff4E)



## Installation
The code was developed in Python 3.7 with PyTorch and transformers 3.1.
The multimodal specific code is in `multimodal_transformers` folder.
```
pip install multimodal-transformers
```

## Included Datasets
This repository also includes two kaggle datasets which contain text data and 
rich tabular features
* [Women's Clothing E-Commerce Reviews](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews) for Recommendation Prediction (Classification)
* [Melbourne Airbnb Open Data](https://www.kaggle.com/tylerx/melbourne-airbnb-open-data) for Price Prediction (Regression)

 

## Working Examples
To quickly see these models in action on say one of the above datasets with preset configurations 
```
$ python main.py ./datasets/Melbourne_Airbnb_Open_Data/train_config.json
```

Or if you prefer command line arguments run 
```
$ python main.py \
    --output_dir=./logs/test \
    --task=classification \
    --combine_feat_method=individual_mlps_on_cat_and_numerical_feats_then_concat \
    --do_train \
    --model_name_or_path=distilbert-base-uncased \
    --data_path=./datasets/Womens_Clothing_E-Commerce_Reviews \
    --column_info_path=./datasets/Womens_Clothing_E-Commerce_Reviews/column_info.json
```
`main.py` expects a `json` file detailing which columns in a dataset contain text, 
categorical, or numerical input features. It also expects a path to the folder where
the data is stored as `train.csv`, and `test.csv`(and if given `val.csv`).For more details on the arguments see 
`multimodal_exp_args.py`.
### Notebook Introduction
To see the modules come together in a notebook: \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/georgianpartners/Multimodal-Toolkit/blob/master/notebooks/text_w_tabular_classification.ipynb)

## Included Methods
| combine feat method |description | requires both cat and num features | 
|:--------------|:-------------------|:-------|
| text_only | Uses just the text columns as processed by a HuggingFace transformer before final classifier layer(s). Essentially equivalent to HuggingFace's `ForSequenceClassification` models |  False | 
| concat | Concatenate transformer output, numerical feats, and categorical feats all at once before final classifier layer(s) | False |
| mlp_on_categorical_then_concat | MLP on categorical feats then concat transformer output, numerical feats, and processed categorical feats before final classifier layer(s) | False (Requires cat feats)
| individual_mlps_on_cat_and_numerical_feats_then_concat | Separate MLPs on categorical feats and numerical feats then concatenation of transformer output, with processed numerical feats, and processed categorical feats before final classifier layer(s). | False
| mlp_on_concatenated_cat_and_numerical_feats_then_concat | MLP on concatenated categorical and numerical feat then concatenated with transformer output before final classifier layer(s) | True
| attention_on_cat_and_numerical_feats | Attention based summation of transformer outputs, numerical feats, and categorical feats queried by transformer outputs before final classifier layer(s). | False
| gating_on_cat_and_num_feats_then_sum | Gated summation of transformer outputs, numerical feats, and categorical feats before final classifier layer(s). Inspired by [Integrating Multimodal Information in Large Pretrained Transformers](https://www.aclweb.org/anthology/2020.acl-main.214.pdf) which performs the mechanism for each token. | False
| weighted_feature_sum_on_transformer_cat_and_numerical_feats | Learnable weighted feature-wise sum of transformer outputs, numerical feats and categorical feats for each feature dimension before final classifier layer(s) | False

## Results
The following tables shows the results on the two included datasets's respective test sets, by running main.py 

### Review Prediction
Specific training parameters can be seen in `datasets/Womens_Clothing_E-Commerce_Reviews/train_config.json`.
Non specified parameters are the default. 

Model | Combine Feat Method |F1 | ROC AUC | PR AUC
--------|-------------|---------|------- | -------
Bert Base Uncased | text_only | 0.959 | 0.969 | 0.993
Bert Base Uncased | individual_mlps_on_cat_and_numerical_feats_then_concat | 0.958 | 0.968 | 0.993
Bert Base Uncased | attention_on_cat_and_numerical_feats | 0.959 | 0.970 | 0.993
Bert Base Uncased | gating_on_cat_and_num_feats_then_sum | 0.961 | **0.976** | **0.995**
Bert Base Uncased | weighted_feature_sum_on_transformer_cat_and_numerical_feats | **0.963** | **0.976** | 0.994


### Pricing Prediction
Specific training parameters can be seen in `datasets/Melbourne_Airbnb_Open_Data/train_config.json`.

Model | Combine Feat Method | MAE | RMSE | 
--------|-------------|---------|------- | 
Bert Base Multilingual Uncased | text_only | 78.77 | 175.93 |
Bert Base Multilingual Uncased | individual_mlps_on_cat_and_numerical_feats_then_concat | 58.58 | **158.69** 
Bert Base Multilingual Uncased | attention_on_cat_and_numerical_feats | 61.10 |160.51
Bert Base Multilingual Uncased | gating_on_cat_and_num_feats_then_sum | **57.56** | 159.22 
Bert Base Multilingual Uncased | weighted_feature_sum_on_transformer_cat_and_numerical_feats | 60.11 | 159.12 
