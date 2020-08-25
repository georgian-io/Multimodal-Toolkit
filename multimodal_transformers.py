from torch import nn
from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
)

from combine_tabular_feat import TabularFeatCombiner
from layer_utils import MLP, calc_mlp_dims, hf_loss_func


class BertWithTabular(BertForSequenceClassification):

    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        self.tabular_combiner = TabularFeatCombiner(hf_model_config)
        combined_feat_dim = self.tabular_combiner.final_out_dim
        self.tabular_config = hf_model_config.tabular_config
        if self.tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                self.tabular_config.num_labels)
        else:
            dims = calc_mlp_dims(combined_feat_dim,
                                 division=self.tabular_config.mlp_division,
                                 output_dim=self.tabular_config.num_labels)
            self.tabular_classifier = MLP(combined_feat_dim,
                                          self.tabular_config.num_labels,
                                          num_hidden_lyr=len(dims),
                                          dropout_prob=self.tabular_config.mlp_dropout,
                                          hidden_channels=dims,
                                          bn=True)

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
                                                              self.tabular_config.num_labels,
                                                              class_weights)
        return loss, logits, classifier_layer_outputs


class RobertaWithTabular(RobertaForSequenceClassification):

    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        self.tabular_combiner = TabularFeatCombiner(hf_model_config)
        combined_feat_dim = self.tabular_combiner.final_out_dim
        self.tabular_config = hf_model_config.tabular_config
        self.dropout = nn.Dropout(hf_model_config.hidden_dropout_prob)
        if self.tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                self.tabular_config.num_labels)
        else:
            dims = calc_mlp_dims(combined_feat_dim,
                                 division=self.tabular_config.mlp_division,
                                 output_dim=self.tabular_config.num_labels)
            self.tabular_classifier = MLP(combined_feat_dim,
                                          self.tabular_config.num_labels,
                                          num_hidden_lyr=len(dims),
                                          dropout_prob=self.tabular_config.mlp_dropout,
                                          hidden_channels=dims,
                                          bn=True)

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
                                                              self.tabular_config.num_labels,
                                                              class_weights)
        return loss, logits, classifier_layer_outputs


class DistilBertWithTabular(DistilBertForSequenceClassification):
    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        self.tabular_combiner = TabularFeatCombiner(hf_model_config)
        combined_feat_dim = self.tabular_combiner.final_out_dim
        self.tabular_config = hf_model_config.tabular_config
        self.dropout = nn.Dropout(hf_model_config.seq_classif_dropout)
        if self.tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                self.tabular_config.num_labels)
        else:
            dims = calc_mlp_dims(combined_feat_dim,
                                 division=self.tabular_config.mlp_division,
                                 output_dim=self.tabular_config.num_labels)
            self.tabular_classifier = MLP(combined_feat_dim,
                                          self.tabular_config.num_labels,
                                          num_hidden_lyr=len(dims),
                                          dropout_prob=self.tabular_config.mlp_dropout,
                                          hidden_channels=dims,
                                          bn=True)

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
                                                              self.tabular_config.num_labels,
                                                              class_weights)
        return loss, logits, classifier_layer_outputs