# Multimodal Transformers | Transformers with Tabular Data

--------------------------------------------------------------------------------
**[Documentation](https://multimodal-toolkit.readthedocs.io/en/latest/index.html)** | **[Colab Notebook](https://multimodal-toolkit.readthedocs.io/en/latest/notes/colab_example.html)** | **[Blog Post](https://medium.com/georgian-impact-blog/how-to-incorporate-tabular-data-with-huggingface-transformers-b70ac45fcfb4)**

A toolkit for incorporating multimodal data on top of text data for classification
and regression tasks. It uses HuggingFace transformers as the base model for text features.
The toolkit adds a combining module that takes the outputs of the transformer in addition to categorical and numerical features
to produce rich multimodal features for downstream classification/regression layers.
Given a pretrained transformer, the parameters of the combining module and transformer are trained based
on the supervised task. For a brief literature review, check out the accompanying [blog post](https://medium.com/georgian-impact-blog/how-to-incorporate-tabular-data-with-huggingface-transformers-b70ac45fcfb4) on Georgian's Impact Blog. 

![](https://drive.google.com/uc?export=view&id=1kyExPDQNkg49NRYgcw2wk8xg4QtQ6Ppt)



## Installation
The code was developed in Python 3.7 with PyTorch and Transformers 4.26.1.
The multimodal specific code is in `multimodal_transformers` folder.
```
pip install multimodal-transformers
```

## Supported Transformers
The following Hugging Face Transformers are supported to handle tabular data. See the documentation [here](https://multimodal-toolkit.readthedocs.io/en/latest/modules/model.html#module-multimodal_transformers.model.tabular_transformers).
* [BERT](https://huggingface.co/transformers/v3.1.0/model_doc/bert.html) from Devlin et al.:
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (ACL 2019)
* [ALBERT](https://huggingface.co/transformers/v3.1.0/model_doc/albert.html) from Lan et al.: [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
](https://arxiv.org/abs/1909.11942) (ICLR 2020)
* [DistilBERT](https://huggingface.co/transformers/v3.1.0/model_doc/distilbert.html) from Sanh et al.: 
[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) (NeurIPS 2019)
* [RoBERTa](https://huggingface.co/transformers/v3.1.0/model_doc/roberta.html) 
from Liu et al.: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
* [XLM](https://huggingface.co/transformers/v3.1.0/model_doc/xlm.html) from Lample et al.: [Cross-lingual Language Model Pretraining
](https://arxiv.org/abs/1901.07291) (NeurIPS 2019)
* [XLNET](https://huggingface.co/transformers/v3.1.0/model_doc/xlnet.html) from Yang et al.:
[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) (NeurIPS 2019)
* [XLM-RoBERTa](https://huggingface.co/transformers/v3.1.0/model_doc/xlmroberta.html) from Conneau et al.:
[Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) (ACL 2020)

## Included Datasets
This repository also includes two kaggle datasets which contain text data and 
rich tabular features
* [Women's Clothing E-Commerce Reviews](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews) for Recommendation Prediction (Classification)
* [Melbourne Airbnb Open Data](https://www.kaggle.com/tylerx/melbourne-airbnb-open-data) for Price Prediction (Regression)
* [PetFindermy Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction) for Pet Adoption Speed Prediction (Multiclass Classification)
 

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
### Simple baseline model
In practice, taking the categorical and numerical features as they are and just tokenizing them and just concatenating them to 
the text columns as extra text sentences is a strong baseline. To do that here, just specify all the categorical and numerical
columns as text columns and set `combine_feat_method` to `text_only`. For example for each of the included sample datasets in `./datasets`, 
in `train_config.json` change `combine_feat_method` to `text_only` and `column_info_path` to  `./datasets/{dataset}/column_info_all_text.json`.

In the experiments below this baseline corresponds to Combine Feat Method being `unimodal`.

## Results
The following tables shows the results on the two included datasets's respective test sets, by running main.py 
Non specified parameters are the default. 

### Review Prediction
Specific training parameters can be seen in `datasets/Womens_Clothing_E-Commerce_Reviews/train_config.json`.

There are **2** text columns, **3** categorical columns, and **3** numerical columns.

Model | Combine Feat Method |F1 | PR AUC
--------|-------------|---------|------- 
Bert Base Uncased | text_only | 0.957 | 0.992
Bert Base Uncased | unimodal | **0.968** | **0.995**
Bert Base Uncased | concat | 0.958 | 0.992
Bert Base Uncased | individual_mlps_on_cat_and_numerical_feats_then_concat | 0.959 | 0.992
Bert Base Uncased | attention_on_cat_and_numerical_feats | 0.959 | 0.992
Bert Base Uncased | gating_on_cat_and_num_feats_then_sum | 0.961 | 0.994
Bert Base Uncased | weighted_feature_sum_on_transformer_cat_and_numerical_feats | 0.962 | 0.994


### Pricing Prediction
Specific training parameters can be seen in `datasets/Melbourne_Airbnb_Open_Data/train_config.json`.

There are **3** text columns, **74** categorical columns, and **15** numerical columns.

Model | Combine Feat Method | MAE | RMSE | 
--------|-------------|---------|------- | 
Bert Base Multilingual Uncased | text_only | 82.74 | 254.0 |
Bert Base Multilingual Uncased | unimodal | 79.34 | 245.2 |
Bert Base Uncased | concat | **65.68** | 239.3 
Bert Base Multilingual Uncased | individual_mlps_on_cat_and_numerical_feats_then_concat | 66.73 | **237.3**  
Bert Base Multilingual Uncased | attention_on_cat_and_numerical_feats | 74.72 |246.3
Bert Base Multilingual Uncased | gating_on_cat_and_num_feats_then_sum | 66.64 | 237.8 
Bert Base Multilingual Uncased | weighted_feature_sum_on_transformer_cat_and_numerical_feats | 71.19 | 245.2 


### Pet Adoption Prediction
Specific training parameters can be seen in `datasets/PetFindermy_Adoption_Prediction`
There are **2** text columns, **14** categorical columns, and **5** numerical columns.

Model | Combine Feat Method | F1_macro | F1_micro | 
--------|-------------|---------|------- | 
Bert Base Multilingual Uncased | text_only | 0.088 | 0.281 |
Bert Base Multilingual Uncased | unimodal | 0.089 | 0.283 |
Bert Base Uncased | concat | 0.199 | 0.362 
Bert Base Multilingual Uncased | individual_mlps_on_cat_and_numerical_feats_then_concat | 0.244 | 0.352
Bert Base Multilingual Uncased | attention_on_cat_and_numerical_feats | 0.254 | 0.375
Bert Base Multilingual Uncased | gating_on_cat_and_num_feats_then_sum | **0.275** | 0.375 
Bert Base Multilingual Uncased | weighted_feature_sum_on_transformer_cat_and_numerical_feats | 0.266 | **0.380**

## Citation
We now have a [paper](https://www.aclweb.org/anthology/2021.maiworkshop-1.10/) you can cite for the Multimodal-Toolkit.
```bibtex
@inproceedings{gu-budhkar-2021-package,
    title = "A Package for Learning on Tabular and Text Data with Transformers",
    author = "Gu, Ken  and
      Budhkar, Akshay",
    booktitle = "Proceedings of the Third Workshop on Multimodal Artificial Intelligence",
    month = jun,
    year = "2021",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.maiworkshop-1.10",
    doi = "10.18653/v1/2021.maiworkshop-1.10",
    pages = "69--73",
}
```
