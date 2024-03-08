# Combine Methods

This page explains the methods that are supported by `multimodal_transformers.tabular_combiner.TabularFeatCombiner`.
See the table for details.

If you have rich categorical and numerical features any of the `attention`, `gating`, or `weighted sum` methods are worth trying. 

The following describes each supported method and whether or not it requires both categorical and numerical features.

| Combine Feat Method | Description | requires both cat and num features | 
|:--------------|:-------------------|:-------|
| text_only | Uses just the text columns as processed by transformer before final classifier layer(s). Essentially equivalent to HuggingFace's `ForSequenceClassification` models |  False | 
| concat | Concatenate transformer output, numerical feats, and categorical feats all at once before final classifier layer(s) | False |
| mlp_on_categorical_then_concat | MLP on categorical feats then concat transformer output, numerical feats, and processed categorical feats before final classifier layer(s) | False (Requires cat feats)
| individual_mlps_on_cat_and_numerical_feats_then_concat | Separate MLPs on categorical feats and numerical feats then concatenation of transformer output, with processed numerical feats, and processed categorical feats before final classifier layer(s). | False
| mlp_on_concatenated_cat_and_numerical_feats_then_concat | MLP on concatenated categorical and numerical feat then concatenated with transformer output before final classifier layer(s) | True
| attention_on_cat_and_numerical_feats | Attention based summation of transformer outputs, numerical feats, and categorical feats queried by transformer outputs before final classifier layer(s). | False
| gating_on_cat_and_num_feats_then_sum | Gated summation of transformer outputs, numerical feats, and categorical feats before final classifier layer(s). Inspired by [Integrating Multimodal Information in Large Pretrained Transformers](https://www.aclweb.org/anthology/2020.acl-main.214.pdf) which performs the mechanism for each token. | False
| weighted_feature_sum_on_transformer_cat_and_numerical_feats | Learnable weighted feature-wise sum of transformer outputs, numerical feats and categorical feats for each feature dimension before final classifier layer(s) | False

This table shows the the equations involved with each method. First we define some notations:

* ![m](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cmathbf%7Bm%7D)  &nbsp; denotes the combined multimodal features
* ![x](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cmathbf%7Bx%7D)  &nbsp; denotes the output text features from the transformer
* ![c](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cmathbf%7Bc%7D)  &nbsp; denotes the categorical features
* ![n](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cmathbf%7Bn%7D)  &nbsp; denotes the numerical features
* ![h_theta](https://latex.codecogs.com/svg.latex?%5Cinline%20h_%7B%5Cmathbf%7B%5CTheta%7D%7D) denotes a MLP parameterized by ![theta](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cmathbf%7B%5CTheta%7D)
* ![W](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BW%7D) &nbsp; denotes a weight matrix
* ![b](https://latex.codecogs.com/svg.latex?b)  &nbsp; denotes a scalar bias

| Combine Feat Method | Equation |
|:--------------|:-------------------|
| text_only | ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bm%7D%20%3D%20%5Cmathbf%7Bx%7D) |
| concat | ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bm%7D%20%3D%20%5Cmathbf%7Bx%7D%20%5C%2C%20%5CVert%20%5C%2C%20%5Cmathbf%7Bc%7D%20%5C%2C%20%5CVert%20%5C%2C%20%5Cmathbf%7Bn%7D) 
| mlp_on_categorical_then_concat | ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bm%7D%20%3D%20%5Cmathbf%7Bx%7D%20%5C%2C%20%5CVert%20%5C%2C%20h_%7B%5Cmathbf%7B%5CTheta%7D%7D%28%20%5Cmathbf%7Bc%7D%29%20%5C%2C%20%5CVert%20%5C%2C%20%5Cmathbf%7Bn%7D)
| individual_mlps_on_cat_and_<br>numerical_feats_then_concat | ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bm%7D%20%3D%20%5Cmathbf%7Bx%7D%20%5C%2C%20%5CVert%20%5C%2C%20h_%7B%5Cmathbf%7B%5CTheta_c%7D%7D%28%20%5Cmathbf%7Bc%7D%29%20%5C%2C%20%5CVert%20%5C%2C%20h_%7B%5Cmathbf%7B%5CTheta_n%7D%7D%28%5Cmathbf%7Bn%7D%29)
| mlp_on_concatenated_cat_and_<br>numerical_feats_then_concat | ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bm%7D%20%3D%20%5Cmathbf%7Bx%7D%20%5C%2C%20%5CVert%20%5C%2C%20h_%7B%5Cmathbf%7B%5CTheta%7D%7D%28%20%5Cmathbf%7Bc%7D%20%5C%2C%20%5CVert%20%5C%2C%20%5Cmathbf%7Bn%7D%29)
| attention_on_cat_and_numerical_feats | ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bm%7D%20%3D%20%5Calpha_%7Bx%2Cx%7D%5Cmathbf%7BW%7D_x%5Cmathbf%7Bx%7D%20&plus;%20%5Calpha_%7Bx%2Cc%7D%5Cmathbf%7BW%7D_c%5Cmathbf%7Bc%7D%20&plus;%20%5Calpha_%7Bx%2Cn%7D%5Cmathbf%7BW%7D_n%5Cmathbf%7Bn%7D) <br><br> where <br><br> ![equation](https://latex.codecogs.com/svg.latex?%5Calpha_%7Bi%2Cj%7D%20%3D%20%5Cfrac%7B%20%5Cexp%5Cleft%28%5Cmathrm%7BLeakyReLU%7D%5Cleft%28%5Cmathbf%7Ba%7D%5E%7B%5Ctop%7D%20%5B%5Cmathbf%7BW%7D_i%5Cmathbf%7Bx%7D_i%20%5C%2C%20%5CVert%20%5C%2C%20%5Cmathbf%7BW%7D_j%5Cmathbf%7Bx%7D_j%5D%20%5Cright%29%5Cright%29%7D%20%7B%5Csum_%7Bk%20%5Cin%20%5C%7B%20x%2C%20c%2C%20n%20%5C%7D%7D%20%5Cexp%5Cleft%28%5Cmathrm%7BLeakyReLU%7D%5Cleft%28%5Cmathbf%7Ba%7D%5E%7B%5Ctop%7D%20%5B%5Cmathbf%7BW%7D_i%5Cmathbf%7Bx%7D_i%20%5C%2C%20%5CVert%20%5C%2C%20%5Cmathbf%7BW%7D_k%5Cmathbf%7Bx%7D_k%5D%20%5Cright%29%5Cright%29%7D.) | 
| gating_on_cat_and_num_feats_<br>then_sum | ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bm%7D%3D%20%5Cmathbf%7Bx%7D%20&plus;%20%5Calpha%5Cmathbf%7Bh%7D) <br><br> ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bh%7D%20%3D%20%5Cmathbf%7Bg_c%7D%20%5Codot%20%28%5Cmathbf%7BW%7D_c%5Cmathbf%7Bc%7D%29%20&plus;%20%5Cmathbf%7Bg_n%7D%20%5Codot%20%28%5Cmathbf%7BW%7D_n%5Cmathbf%7Bn%7D%29%20&plus;%20b_h) <br><br> ![equation](https://latex.codecogs.com/svg.latex?%5Calpha%20%3D%20%5Cmathrm%7Bmin%7D%28%20%5Cfrac%7B%5C%7C%20%5Cmathbf%7Bx%7D%20%5C%7C_2%7D%7B%5C%7C%20%5Cmathbf%7Bh%7D%20%5C%7C_2%7D*%5Cbeta%2C%201%29) <br><br> ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bg%7D_i%20%3D%20R%28%5Cmathbf%7BW%7D_%7Bgi%7D%5B%5Cmathbf%7Bi%7D%20%5C%2C%20%5CVert%20%5C%2C%20%5Cmathbf%7Bx%7D%5D&plus;%20b_i%29) <br><br> where ![equation](https://latex.codecogs.com/svg.latex?%5Cbeta) is a hyperparameter and  ![equation](https://latex.codecogs.com/svg.latex?R) is an activation function 
| weighted_feature_sum_on_transformer_<br>cat_and_numerical_feats | ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bm%7D%20%3D%20%5Cmathbf%7Bx%7D%20&plus;%20%5Cmathbf%7BW%7D_%7Bc%27%7D%20%5Codot%20%5Cmathbf%7BW%7D_c%20%5Cmathbf%7Bc%7D%20&plus;%20%5Cmathbf%7BW%7D_%7Bn%27%7D%20%5Codot%20%5Cmathbf%7BW%7D_n%20%5Cmathbf%7Bn%7D)
