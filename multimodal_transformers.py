from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertForSequenceClassification, BertModel, BertPreTrainedModel

from combine_tabular_feat import TabularFeatCombiner
from layer_utils import MLP, calc_mlp_dims, hf_loss_func


class BertWithTabular(BertForSequenceClassification):

    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        self.tabular_combiner = TabularFeatCombiner(hf_model_config)
        combined_feat_dim = self.tabular_combiner.final_out_dim
        tabular_config = hf_model_config.tabular_config
        if tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                hf_model_config.num_labels)
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


