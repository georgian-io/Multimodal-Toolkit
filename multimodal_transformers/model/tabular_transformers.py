from torch import nn
from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
)
from transformers.modeling_bert import BERT_INPUTS_DOCSTRING
from transformers.modeling_roberta import ROBERTA_INPUTS_DOCSTRING
from transformers.modeling_distilbert import DISTILBERT_INPUTS_DOCSTRING
from transformers.file_utils import add_start_docstrings_to_callable

from .tabular_combiner import TabularFeatCombiner
from .tabular_config import TabularConfig
from .layer_utils import MLP, calc_mlp_dims, hf_loss_func


class BertWithTabular(BertForSequenceClassification):
    """
    Bert Model transformer with a sequence classification/regression head as well as
    a TabularFeatCombiner module to combine categorical and numerical features
    with the Bert pooled output

    Parameters:
        hf_model_config (:class:`~transformers.BertConfig`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            :obj:`TabularConfig` instance specifying the configs for :obj:`TabularFeatCombiner`
    """

    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        tabular_config = hf_model_config.tabular_config
        if type(tabular_config) is dict:  # when loading from saved model
            tabular_config = TabularConfig(**tabular_config)
        else:
            self.config.tabular_config = tabular_config.__dict__

        tabular_config.text_feat_dim = hf_model_config.hidden_size
        tabular_config.hidden_dropout_prob = hf_model_config.hidden_dropout_prob
        self.tabular_combiner = TabularFeatCombiner(tabular_config)
        self.num_labels = tabular_config.num_labels
        combined_feat_dim = self.tabular_combiner.final_out_dim
        if tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                tabular_config.num_labels)
        else:
            dims = calc_mlp_dims(combined_feat_dim,
                                 division=tabular_config.mlp_division,
                                 output_dim=tabular_config.num_labels)
            self.tabular_classifier = MLP(combined_feat_dim,
                                          tabular_config.num_labels,
                                          num_hidden_lyr=len(dims),
                                          dropout_prob=tabular_config.mlp_dropout,
                                          hidden_channels=dims,
                                          bn=True)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        class_weights=None,
        output_attentions=None,
        output_hidden_states=None,
        cat_feats=None,
        numerical_feats=None
    ):
        r"""
        class_weights (:obj:`torch.FloatTensor` of shape :obj:`(tabular_config.num_labels,)`, `optional`, defaults to :obj:`None`):
            Class weights to be used for cross entropy loss function for classification task
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`tabular_config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`tabular_config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        cat_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.cat_feat_dim)`, `optional`, defaults to :obj:`None`):
            Categorical features to be passed in to the TabularFeatCombiner
        numerical_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.numerical_feat_dim)`, `optional`, defaults to :obj:`None`):
            Numerical features to be passed in to the TabularFeatCombiner
    Returns:
        :obj:`tuple` comprising various elements depending on configuration and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if tabular_config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.num_labels)`):
            Classification (or regression if tabular_config.num_labels==1) scores (before SoftMax).
        classifier_layer_outputs(:obj:`list` of :obj:`torch.FloatTensor`):
            The outputs of each layer of the final classification layers. The 0th index of this list is the
            combining module's output
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        combined_feats = self.tabular_combiner(pooled_output,
                                               cat_feats,
                                               numerical_feats)
        loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
                                                              self.tabular_classifier,
                                                              labels,
                                                              self.num_labels,
                                                              class_weights)
        return loss, logits, classifier_layer_outputs


class RobertaWithTabular(RobertaForSequenceClassification):
    """
    Roberta Model transformer with a sequence classification/regression head as well as
    a TabularFeatCombiner module to combine categorical and numerical features
    with the Roberta pooled output

    Parameters:
        hf_model_config (:class:`~transformers.RobertaConfig`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            :obj:`TabularConfig` instance specifying the configs for :obj:`TabularFeatCombiner`
    """
    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        tabular_config = hf_model_config.tabular_config
        if type(tabular_config) is dict:  # when loading from saved model
            tabular_config = TabularConfig(**tabular_config)
        else:
            self.config.tabular_config = tabular_config.__dict__

        tabular_config.text_feat_dim = hf_model_config.hidden_size
        tabular_config.hidden_dropout_prob = hf_model_config.hidden_dropout_prob
        self.tabular_combiner = TabularFeatCombiner(tabular_config)
        self.num_labels = tabular_config.num_labels
        combined_feat_dim = self.tabular_combiner.final_out_dim
        self.dropout = nn.Dropout(hf_model_config.hidden_dropout_prob)
        if tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                tabular_config.num_labels)
        else:
            dims = calc_mlp_dims(combined_feat_dim,
                                 division=tabular_config.mlp_division,
                                 output_dim=tabular_config.num_labels)
            self.tabular_classifier = MLP(combined_feat_dim,
                                          tabular_config.num_labels,
                                          num_hidden_lyr=len(dims),
                                          dropout_prob=tabular_config.mlp_dropout,
                                          hidden_channels=dims,
                                          bn=True)

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        class_weights=None,
        cat_feats=None,
        numerical_feats=None
    ):
        r"""
        class_weights (:obj:`torch.FloatTensor` of shape :obj:`(tabular_config.num_labels,)`, `optional`, defaults to :obj:`None`):
            Class weights to be used for cross entropy loss function for classification task
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`tabular_config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`tabular_config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        cat_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.cat_feat_dim)`, `optional`, defaults to :obj:`None`):
            Categorical features to be passed in to the TabularFeatCombiner
        numerical_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.numerical_feat_dim)`, `optional`, defaults to :obj:`None`):
            Numerical features to be passed in to the TabularFeatCombiner

    Returns:
        :obj:`tuple` comprising various elements depending on configuration and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if tabular_config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.num_labels)`):
            Classification (or regression if tabular_config.num_labels==1) scores (before SoftMax).
        classifier_layer_outputs(:obj:`list` of :obj:`torch.FloatTensor`):
            The outputs of each layer of the final classification layers. The 0th index of this list is the
            combining module's output

        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        text_feats = sequence_output[:, 0, :]
        text_feats = self.dropout(text_feats)
        combined_feats = self.tabular_combiner(text_feats,
                                               cat_feats,
                                               numerical_feats)
        loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
                                                              self.tabular_classifier,
                                                              labels,
                                                              self.num_labels,
                                                              class_weights)
        return loss, logits, classifier_layer_outputs


class DistilBertWithTabular(DistilBertForSequenceClassification):
    """
    DistilBert Model transformer with a sequence classification/regression head as well as
    a TabularFeatCombiner module to combine categorical and numerical features
    with the Roberta pooled output

    Parameters:
        hf_model_config (:class:`~transformers.DistilBertConfig`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            :obj:`TabularConfig` instance specifying the configs for :obj:`TabularFeatCombiner`
    """
    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        tabular_config = hf_model_config.tabular_config
        if type(tabular_config) is dict:  # when loading from saved model
            tabular_config = TabularConfig(**tabular_config)
        else:
            self.config.tabular_config = tabular_config.__dict__

        tabular_config.text_feat_dim = hf_model_config.hidden_size
        tabular_config.hidden_dropout_prob = hf_model_config.seq_classif_dropout
        self.tabular_combiner = TabularFeatCombiner(tabular_config)
        self.num_labels = tabular_config.num_labels
        combined_feat_dim = self.tabular_combiner.final_out_dim
        if tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                tabular_config.num_labels)
        else:
            dims = calc_mlp_dims(combined_feat_dim,
                                 division=tabular_config.mlp_division,
                                 output_dim=tabular_config.num_labels)
            self.tabular_classifier = MLP(combined_feat_dim,
                                          tabular_config.num_labels,
                                          num_hidden_lyr=len(dims),
                                          dropout_prob=tabular_config.mlp_dropout,
                                          hidden_channels=dims,
                                          bn=True)

    @add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        class_weights=None,
        cat_feats=None,
        numerical_feats=None
    ):
        r"""
        class_weights (:obj:`torch.FloatTensor` of shape :obj:`(tabular_config.num_labels,)`,`optional`, defaults to :obj:`None`):
            Class weights to be used for cross entropy loss function for classification task
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`tabular_config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`tabular_config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        cat_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.cat_feat_dim)`,`optional`, defaults to :obj:`None`):
            Categorical features to be passed in to the TabularFeatCombiner
        numerical_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.numerical_feat_dim)`,`optional`, defaults to :obj:`None`):
            Numerical features to be passed in to the TabularFeatCombiner
    Returns:
        :obj:`tuple` comprising various elements depending on configuration and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if tabular_config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.num_labels)`):
            Classification (or regression if tabular_config.num_labels==1) scores (before SoftMax).
        classifier_layer_outputs(:obj:`list` of :obj:`torch.FloatTensor`):
            The outputs of each layer of the final classification layers. The 0th index of this list is the
            combining module's output
        """

        distilbert_output = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        text_feats = self.dropout(pooled_output)
        combined_feats = self.tabular_combiner(text_feats,
                                               cat_feats,
                                               numerical_feats)
        loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
                                                              self.tabular_classifier,
                                                              labels,
                                                              self.num_labels,
                                                              class_weights)
        return loss, logits, classifier_layer_outputs