Introduction by Example
=======================

This guide covers how to use the PyTorch module bert_w_tabular in your own project. It shows

.. contents::
    :local:


How to Initialize Transformer With Tabular Modules
---------------------------------------------------
The models which support tabular features are located in :obj:`multimodal_transformers.tabular_transformers`.
These adapted transformer modules expect the same transformer config instances as
the ones from HuggingFace. However, expect a :class:`multimodal_transformers.tabular_config.TabularConfig` instance specifying
the configs.

Say for example we had categorical features of dim 9 and numerical features of dim 5.

.. code-block:: python

    from transformers import BertConfig

    from multimodal_transformers.model.tabular_transformers import BertWithTabular
    from multimodal_transformers.model.tabular_config import TabularConfig

    bert_config = BertConfig.from_pretrained('bert-base-uncased')

    tabular_config = TabularConfig(
            combine_feat_method='attention_on_cat_and_numerical_feats',  # change this to specify the method of combining tabular data
            cat_feat_dim=9,  # need to specify this
            numerical_feat_dim=5,  # need to specify this
            num_labels=2,   # need to specify this, assuming our task is binary classification
            use_num_bn=False,
    )

    bert_config.tabular_config = tabular_config

    model = BertWithTabular.from_pretrained('bert-base-uncased', config=bert_config)


In fact for any HuggingFace transformer model supported in :obj:`multimodal_transformers.model.tabular_transformers` we
can initialize it using :obj:`multimodal_transformers.model.tabular_modelling_auto.AutoModelWithTabular` to
leverage any community trained transformer models

.. code-block:: python

    from transformers import AutoConfig

    from multimodal_transformers.model.tabular_modelling_auto import AutoModelWithTabular
    from multimodal_transformers.model.tabular_config import TabularConfig

    hf_config = AutoConfig.from_pretrained('ipuneetrathore/bert-base-cased-finetuned-finBERT')
    tabular_config = TabularConfig(
            combine_feat_method='attention_on_cat_and_numerical_feats',  # change this to specify the method of combining tabular data
            cat_feat_dim=9,  # need to specify this
            numerical_feat_dim=5,  # need to specify this
            num_labels=2,   # need to specify this, assuming our task is binary classification
    )
    hf_config.tabular_config = tabular_config

    model = AutoModelWithTabular.from_pretrained('ipuneetrathore/bert-base-cased-finetuned-finBERT', config=hf_config)


Foward Pass of Transformer with Tabular Models
----------------------------------------------

During the forward pass we pass HuggingFace's normal `transformer inputs <https://huggingface.co/transformers/glossary.html>`_
as well as our categorical and numerical features.

The forward pass returns

- :obj:`torch.FloatTensor` of shape :obj:`(1,)`: The classification (or regression if tabular_config.num_labels==1) loss
- :obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.num_labels)`: The classification (or regression if tabular_config.num_labels==1) scores (before SoftMax)
- :obj:`list` of :obj:`torch.FloatTensor` The outputs of each layer of the final classification layers. The 0th index of this list is the
  combining module's output

The following example shows a forward pass on two data examples

.. code-block:: python

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    text_1 = "HuggingFace is based in NYC"
    text_2 = "Where is HuggingFace based?"
    model_inputs = tokenizer([text1, text2]

    # 5 numerical features
    numerical_feat = torch.rand(2, 5).float()
    # 9 categorical features
    categorical_feat = torch.tensor([[0, 0, 0, 1, 0, 1, 0, 1, 0],
                                     [1, 0, 0, 0, 1, 0, 1, 0, 0]]).float()
    labels = torch.tensor([1, 0])

    model_inputs['cat_feats'] = categorical_feat
    model_inputs['num_feats'] = numerical_feat
    model_inputs['labels'] = labels

    loss, logits, layer_outs = model(**model_inputs)

We can also pass in the arguments explicitly

.. code-block:: python

    loss, logits, layer_outs = model(
        model_inputs['input_ids'],
        token_type_ids=model_inputs['token_type_ids'],
        labels=labels,
        cat_feats=categorical_feat,
        numerical_feats=numerical_feat
    )




Modifications: Only One Type of Tabular Feature or No Tabular Features
-------------------------------------------------------------------------
If there are no tabular features, the models basically default to the ForSequenceClassification
models from HuggingFace. We must specify :obj:`combine_feat_method='text_only'` in
:class:`multimodal_transformers.model.tabular_config.TabularConfig`. During the forward pass
we can simply pass the text related inputs

.. code-block:: python

    loss, logits, layer_outs = model(
        model_inputs['input_ids'],
        token_type_ids=model_inputs['token_type_ids'],
        labels=labels,
    )

If only one of the features is available, we first must specify a
:obj:`combine_feat_method` that supports only one type of feature available.
See supported methods for more details.
When initializing our tabular config we specify the dimensions of the feature we have.
For example if we only have categorical features

.. code-block:: python

    tabular_config = TabularConfig(
        combine_feat_method='attention_on_cat_and_numerical_feats',  # change this to specify the method of combining tabular data
        cat_feat_dim=9,  # need to specify this
        num_labels=2,   # need to specify this, assuming our task is binary classification
    )

During the forward pass, we also pass only the tabular data that we have.

.. code-block:: python

    loss, logits, layer_outs = model(
        model_inputs['input_ids'],
        token_type_ids=model_inputs['token_type_ids'],
        labels=labels,
        cat_feats=categorical_feat,
    )

Inference
------------
During inference we do not need to pass the labels and we can take the logits from the second output from the forward pass of the model.

.. code-block:: python

    with torch.no_grad():
        _, logits, classifier_outputs = model(
            model_inputs['input_ids'],
            token_type_ids=model_inputs['token_type_ids'],
            cat_feats=categorical_feat,
            numerical_feats=numerical_feat
        )