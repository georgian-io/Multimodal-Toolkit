import torch
from torch import nn
import torch.nn.functional as F

from .layer_utils import calc_mlp_dims, create_act, glorot, zeros, MLP


class TabularFeatCombiner(nn.Module):
    r"""
     Combiner module for combining text features with categorical and numerical features
     The methods of combining, specified by :obj:`tabular_config.combine_feat_method` are shown below.
     :math:`\mathbf{m}` denotes the combined multimodal features,
     :math:`\mathbf{x}` denotes the output text features from the transformer,
     :math:`\mathbf{c}` denotes the categorical features, :math:`\mathbf{t}` denotes the numerical features,
     :math:`h_{\mathbf{\Theta}}` denotes a MLP parameterized by :math:`\Theta`, :math:`W` denotes a weight matrix,
     and :math:`b` denotes a scalar bias

     - **text_only**

         .. math::
             \mathbf{m} = \mathbf{x}

     - **concat**

         .. math::
             \mathbf{m} = \mathbf{x} \, \Vert \, \mathbf{c} \, \Vert \, \mathbf{n}

     - **mlp_on_categorical_then_concat**

         .. math::
             \mathbf{m} = \mathbf{x} \, \Vert \, h_{\mathbf{\Theta}}( \mathbf{c}) \, \Vert \, \mathbf{n}

     - **individual_mlps_on_cat_and_numerical_feats_then_concat**

         .. math::
             \mathbf{m} = \mathbf{x} \, \Vert \, h_{\mathbf{\Theta_c}}( \mathbf{c}) \, \Vert \, h_{\mathbf{\Theta_n}}(\mathbf{n})

     - **mlp_on_concatenated_cat_and_numerical_feats_then_concat**

         .. math::
             \mathbf{m} = \mathbf{x} \, \Vert \, h_{\mathbf{\Theta}}( \mathbf{c} \, \Vert \, \mathbf{n})

     - **attention_on_cat_and_numerical_feats** self attention on the text features

         .. math::
             \mathbf{m} = \alpha_{x,x}\mathbf{W}_x\mathbf{x} + \alpha_{x,c}\mathbf{W}_c\mathbf{c} + \alpha_{x,n}\mathbf{W}_n\mathbf{n}

       where :math:`\mathbf{W}_x` is of shape :obj:`(out_dim, text_feat_dim)`,
       :math:`\mathbf{W}_c` is of shape :obj:`(out_dim, cat_feat_dim)`,
       :math:`\mathbf{W}_n` is of shape :obj:`(out_dim, num_feat_dim)`, and the attention coefficients :math:`\alpha_{i,j}` are computed as

         .. math::
             \alpha_{i,j} =
             \frac{
             \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
             [\mathbf{W}_i\mathbf{x}_i \, \Vert \, \mathbf{W}_j\mathbf{x}_j]
             \right)\right)}
             {\sum_{k \in \{ x, c, n \}}
             \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
             [\mathbf{W}_i\mathbf{x}_i \, \Vert \, \mathbf{W}_k\mathbf{x}_k]
             \right)\right)}.

     - **gating_on_cat_and_num_feats_then_sum** sum of features gated by text features. Inspired by the gating mechanism introduced in `Integrating Multimodal Information in Large Pretrained Transformers <https://www.aclweb.org/anthology/2020.acl-main.214.pdf>`_

         .. math::
             \mathbf{m}= \mathbf{x} + \alpha\mathbf{h}
         .. math::
             \mathbf{h} = \mathbf{g_c} \odot (\mathbf{W}_c\mathbf{c}) + \mathbf{g_n} \odot (\mathbf{W}_n\mathbf{n}) + b_h
         .. math::
             \alpha = \mathrm{min}( \frac{\| \mathbf{x} \|_2}{\| \mathbf{h} \|_2}*\beta, 1)

       where :math:`\beta` is a hyperparamter, :math:`\mathbf{W}_c` is of shape :obj:`(out_dim, cat_feat_dim)`,
       :math:`\mathbf{W}_n` is of shape :obj:`(out_dim, num_feat_dim)`. and the gating vector :math:`\mathbf{g}_i` with activation function :math:`R` is defined as

         .. math::
             \mathbf{g}_i = R(\mathbf{W}_{gi}[\mathbf{i} \, \Vert \, \mathbf{x}]+ b_i)

       where :math:`\mathbf{W}_{gi}` is of shape :obj:`(out_dim, i_feat_dim + text_feat_dim)`

     - **weighted_feature_sum_on_transformer_cat_and_numerical_feats**

         .. math::
             \mathbf{m} = \mathbf{x} + \mathbf{W}_{c'} \odot \mathbf{W}_c \mathbf{c} + \mathbf{W}_{n'} \odot \mathbf{W}_n \mathbf{t}

    Parameters:
        tabular_config (:class:`~multimodal_config.TabularConfig`):
            Tabular model configuration class with all the parameters of the model.

    """

    def __init__(self, tabular_config):
        super().__init__()
        self.combine_feat_method = tabular_config.combine_feat_method
        self.cat_feat_dim = tabular_config.cat_feat_dim
        self.numerical_feat_dim = tabular_config.numerical_feat_dim
        self.num_labels = tabular_config.num_labels
        self.numerical_bn = tabular_config.numerical_bn
        self.categorical_bn = tabular_config.categorical_bn
        self.mlp_act = tabular_config.mlp_act
        self.mlp_dropout = tabular_config.mlp_dropout
        self.mlp_division = tabular_config.mlp_division
        self.text_out_dim = tabular_config.text_feat_dim
        self.tabular_config = tabular_config

        if self.combine_feat_method == "text_only":
            self.final_out_dim = self.text_out_dim
        elif self.combine_feat_method == "concat":
            self.final_out_dim = (
                self.text_out_dim + self.cat_feat_dim + self.numerical_feat_dim
            )
        elif self.combine_feat_method == "mlp_on_categorical_then_concat":
            assert self.cat_feat_dim != 0, "dimension of cat feats should not be 0"
            # reduce dim of categorical features to same of num dim or text dim if necessary
            output_dim = min(
                self.text_out_dim,
                max(
                    self.numerical_feat_dim,
                    self.cat_feat_dim // (self.mlp_division // 2),
                ),
            )
            dims = calc_mlp_dims(self.cat_feat_dim, self.mlp_division, output_dim)
            self.cat_mlp = MLP(
                self.cat_feat_dim,
                output_dim,
                act=self.mlp_act,
                num_hidden_lyr=len(dims),
                dropout_prob=self.mlp_dropout,
                hidden_channels=dims,
                return_layer_outs=False,
                bn=self.categorical_bn,
            )
            self.final_out_dim = (
                self.text_out_dim + output_dim + self.numerical_feat_dim
            )
        elif (
            self.combine_feat_method
            == "mlp_on_concatenated_cat_and_numerical_feats_then_concat"
        ):
            assert self.cat_feat_dim != 0, "dimension of cat feats should not be 0"
            assert (
                self.numerical_feat_dim != 0
            ), "dimension of numerical feats should not be 0"
            output_dim = min(
                self.numerical_feat_dim, self.cat_feat_dim, self.text_out_dim
            )
            in_dim = self.cat_feat_dim + self.numerical_feat_dim
            dims = calc_mlp_dims(in_dim, self.mlp_division, output_dim)
            self.cat_and_numerical_mlp = MLP(
                in_dim,
                output_dim,
                act=self.mlp_act,
                num_hidden_lyr=len(dims),
                dropout_prob=self.mlp_dropout,
                hidden_channels=dims,
                return_layer_outs=False,
                bn=self.categorical_bn and self.numerical_bn,
            )
            self.final_out_dim = self.text_out_dim + output_dim
        elif (
            self.combine_feat_method
            == "individual_mlps_on_cat_and_numerical_feats_then_concat"
        ):
            output_dim_cat = 0
            if self.cat_feat_dim > 0:
                output_dim_cat = max(
                    self.cat_feat_dim // (self.mlp_division // 2),
                    self.numerical_feat_dim,
                )
                dims = calc_mlp_dims(
                    self.cat_feat_dim, self.mlp_division, output_dim_cat
                )
                self.cat_mlp = MLP(
                    self.cat_feat_dim,
                    output_dim_cat,
                    act=self.mlp_act,
                    num_hidden_lyr=len(dims),
                    dropout_prob=self.mlp_dropout,
                    hidden_channels=dims,
                    return_layer_outs=False,
                    bn=self.categorical_bn,
                )

            output_dim_num = 0
            if self.numerical_feat_dim > 0:
                output_dim_num = self.numerical_feat_dim // (self.mlp_division // 2)
                self.num_mlp = MLP(
                    self.numerical_feat_dim,
                    output_dim_num,
                    act=self.mlp_act,
                    dropout_prob=self.mlp_dropout,
                    num_hidden_lyr=1,
                    return_layer_outs=False,
                    bn=self.numerical_bn,
                )
            self.final_out_dim = self.text_out_dim + output_dim_num + output_dim_cat
        elif (
            self.combine_feat_method
            == "weighted_feature_sum_on_transformer_cat_and_numerical_feats"
        ):
            assert (
                self.cat_feat_dim + self.numerical_feat_dim != 0
            ), "should have some non text features"
            if self.cat_feat_dim > 0:
                output_dim_cat = self.text_out_dim
                if self.cat_feat_dim > self.text_out_dim:
                    dims = calc_mlp_dims(
                        self.cat_feat_dim,
                        division=self.mlp_division,
                        output_dim=output_dim_cat,
                    )
                    self.cat_layer = MLP(
                        self.cat_feat_dim,
                        output_dim_cat,
                        act=self.mlp_act,
                        num_hidden_lyr=len(dims),
                        dropout_prob=self.mlp_dropout,
                        hidden_channels=dims,
                        return_layer_outs=False,
                        bn=self.categorical_bn,
                    )
                else:
                    self.cat_layer = nn.Linear(self.cat_feat_dim, output_dim_cat)
                self.dropout_cat = nn.Dropout(self.mlp_dropout)
                self.weight_cat = nn.Parameter(torch.rand(output_dim_cat))
            if self.numerical_feat_dim > 0:
                output_dim_num = self.text_out_dim
                if self.numerical_feat_dim > self.text_out_dim:
                    dims = calc_mlp_dims(
                        self.numerical_feat_dim,
                        division=self.mlp_division,
                        output_dim=output_dim_num,
                    )
                    self.num_layer = MLP(
                        self.numerical_feat_dim,
                        output_dim_num,
                        act=self.mlp_act,
                        num_hidden_lyr=len(dims),
                        dropout_prob=self.mlp_dropout,
                        hidden_channels=dims,
                        return_layer_outs=False,
                        bn=self.numerical_bn,
                    )
                else:
                    self.num_layer = nn.Linear(self.numerical_feat_dim, output_dim_num)
                self.dropout_num = nn.Dropout(self.mlp_dropout)
                self.weight_num = nn.Parameter(torch.rand(output_dim_num))

            self.act_func = create_act(self.mlp_act)
            self.layer_norm = nn.LayerNorm(self.text_out_dim)
            self.final_dropout = nn.Dropout(tabular_config.hidden_dropout_prob)
            self.final_out_dim = self.text_out_dim

        elif self.combine_feat_method == "attention_on_cat_and_numerical_feats":
            assert (
                self.cat_feat_dim + self.numerical_feat_dim != 0
            ), "should have some non-text features for this method"

            output_dim = self.text_out_dim
            if self.cat_feat_dim > 0:
                if self.cat_feat_dim > self.text_out_dim:
                    output_dim_cat = self.text_out_dim
                    dims = calc_mlp_dims(
                        self.cat_feat_dim,
                        division=self.mlp_division,
                        output_dim=output_dim_cat,
                    )
                    self.cat_mlp = MLP(
                        self.cat_feat_dim,
                        output_dim_cat,
                        num_hidden_lyr=len(dims),
                        dropout_prob=self.mlp_dropout,
                        return_layer_outs=False,
                        hidden_channels=dims,
                        bn=self.categorical_bn,
                    )
                else:
                    output_dim_cat = self.cat_feat_dim
                self.weight_cat = nn.Parameter(torch.rand((output_dim_cat, output_dim)))
                self.bias_cat = nn.Parameter(torch.zeros(output_dim))

            if self.numerical_feat_dim > 0:
                if self.numerical_feat_dim > self.text_out_dim:
                    output_dim_num = self.text_out_dim
                    dims = calc_mlp_dims(
                        self.numerical_feat_dim,
                        division=self.mlp_division,
                        output_dim=output_dim_num,
                    )
                    self.num_mlp = MLP(
                        self.numerical_feat_dim,
                        output_dim_num,
                        num_hidden_lyr=len(dims),
                        dropout_prob=self.mlp_dropout,
                        return_layer_outs=False,
                        hidden_channels=dims,
                        bn=self.numerical_bn,
                    )
                else:
                    output_dim_num = self.numerical_feat_dim
                self.weight_num = nn.Parameter(torch.rand((output_dim_num, output_dim)))
                self.bias_num = nn.Parameter(torch.zeros(output_dim))

            self.weight_transformer = nn.Parameter(
                torch.rand(self.text_out_dim, output_dim)
            )
            self.weight_a = nn.Parameter(torch.rand((1, output_dim + output_dim)))
            self.bias_transformer = nn.Parameter(torch.rand(output_dim))
            self.bias = nn.Parameter(torch.zeros(output_dim))
            self.negative_slope = 0.2
            self.final_out_dim = output_dim
            self.__reset_parameters()
        elif self.combine_feat_method == "gating_on_cat_and_num_feats_then_sum":
            self.act_func = create_act(self.mlp_act)
            if self.cat_feat_dim > 0:
                if self.cat_feat_dim > self.text_out_dim:
                    dims = calc_mlp_dims(
                        self.numerical_feat_dim,
                        division=self.mlp_division,
                        output_dim=self.text_out_dim,
                    )
                    self.cat_layer = MLP(
                        self.cat_feat_dim,
                        self.text_out_dim,
                        act=self.mlp_act,
                        num_hidden_lyr=len(dims),
                        dropout_prob=self.mlp_dropout,
                        hidden_channels=dims,
                        return_layer_outs=False,
                        bn=self.categorical_bn,
                    )
                self.g_cat_layer = nn.Linear(
                    self.text_out_dim + min(self.text_out_dim, self.cat_feat_dim),
                    self.text_out_dim,
                )
                self.dropout_cat = nn.Dropout(self.mlp_dropout)
                self.h_cat_layer = nn.Linear(
                    min(self.text_out_dim, self.cat_feat_dim),
                    self.text_out_dim,
                    bias=False,
                )
            if self.numerical_feat_dim > 0:
                if self.numerical_feat_dim > self.text_out_dim:
                    dims = calc_mlp_dims(
                        self.numerical_feat_dim,
                        division=self.mlp_division,
                        output_dim=self.text_out_dim,
                    )
                    self.num_layer = MLP(
                        self.numerical_feat_dim,
                        self.text_out_dim,
                        act=self.mlp_act,
                        num_hidden_lyr=len(dims),
                        dropout_prob=self.mlp_dropout,
                        hidden_channels=dims,
                        return_layer_outs=False,
                        bn=self.numerical_bn,
                    )
                self.g_num_layer = nn.Linear(
                    min(self.numerical_feat_dim, self.text_out_dim) + self.text_out_dim,
                    self.text_out_dim,
                )
                self.dropout_num = nn.Dropout(self.mlp_dropout)
                self.h_num_layer = nn.Linear(
                    min(self.text_out_dim, self.numerical_feat_dim),
                    self.text_out_dim,
                    bias=False,
                )
            self.h_bias = nn.Parameter(torch.zeros(self.text_out_dim))
            self.layer_norm = nn.LayerNorm(self.text_out_dim)
            self.final_out_dim = self.text_out_dim
        else:
            raise ValueError(
                f"combine_feat_method {self.combine_feat_method} " f"not implemented"
            )

    def forward(self, text_feats, cat_feats=None, numerical_feats=None):
        """
        Args:
            text_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, text_out_dim)`):
                The tensor of text features. This is assumed to be the output from a HuggingFace transformer model
            cat_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, cat_feat_dim)`, `optional`, defaults to :obj:`None`)):
                The tensor of categorical features
            numerical_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, numerical_feat_dim)`, `optional`, defaults to :obj:`None`):
                The tensor of numerical features
        Returns:
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, final_out_dim)`:
                A tensor representing the combined features

        """
        if cat_feats is None:
            cat_feats = torch.zeros((text_feats.shape[0], 0)).to(text_feats.device)
        if numerical_feats is None:
            numerical_feats = torch.zeros((text_feats.shape[0], 0)).to(
                text_feats.device
            )

        if self.combine_feat_method == "text_only":
            combined_feats = text_feats
        if self.combine_feat_method == "concat":
            combined_feats = torch.cat((text_feats, cat_feats, numerical_feats), dim=1)
        elif self.combine_feat_method == "mlp_on_categorical_then_concat":
            cat_feats = self.cat_mlp(cat_feats)
            combined_feats = torch.cat((text_feats, cat_feats, numerical_feats), dim=1)
        elif (
            self.combine_feat_method
            == "mlp_on_concatenated_cat_and_numerical_feats_then_concat"
        ):
            tabular_feats = torch.cat((cat_feats, numerical_feats), dim=1)
            tabular_feats = self.cat_and_numerical_mlp(tabular_feats)
            combined_feats = torch.cat((text_feats, tabular_feats), dim=1)
        elif (
            self.combine_feat_method
            == "individual_mlps_on_cat_and_numerical_feats_then_concat"
        ):
            if cat_feats.shape[1] != 0:
                cat_feats = self.cat_mlp(cat_feats)
            if numerical_feats.shape[1] != 0:
                numerical_feats = self.num_mlp(numerical_feats)
            combined_feats = torch.cat((text_feats, cat_feats, numerical_feats), dim=1)
        elif (
            self.combine_feat_method
            == "weighted_feature_sum_on_transformer_cat_and_numerical_feats"
        ):
            if cat_feats.shape[1] != 0:
                cat_feats = self.dropout_cat(self.cat_layer(cat_feats))
                cat_feats = self.weight_cat.expand_as(cat_feats) * cat_feats
            else:
                cat_feats = 0

            if numerical_feats.shape[1] != 0:
                numerical_feats = self.dropout_num(self.num_layer(numerical_feats))
                numerical_feats = (
                    self.weight_num.expand_as(numerical_feats) * numerical_feats
                )
            else:
                numerical_feats = 0
            combined_feats = text_feats + cat_feats + numerical_feats
        elif self.combine_feat_method == "attention_on_cat_and_numerical_feats":
            # attention keyed by transformer text features
            w_text = torch.mm(text_feats, self.weight_transformer)
            g_text = (
                (torch.cat([w_text, w_text], dim=-1) * self.weight_a)
                .sum(dim=1)
                .unsqueeze(0)
                .T
            )

            if cat_feats.shape[1] != 0:
                if self.cat_feat_dim > self.text_out_dim:
                    cat_feats = self.cat_mlp(cat_feats)
                w_cat = torch.mm(cat_feats, self.weight_cat) 
                g_cat = (
                    (torch.cat([w_text, w_cat], dim=-1) * self.weight_a)
                    .sum(dim=1)
                    .unsqueeze(0)
                    .T
                )
            else:
                w_cat = None
                g_cat = torch.zeros(0, device=g_text.device)

            if numerical_feats.shape[1] != 0:
                if self.numerical_feat_dim > self.text_out_dim:
                    numerical_feats = self.num_mlp(numerical_feats)
                w_num = torch.mm(numerical_feats, self.weight_num)
                g_num = (
                    (torch.cat([w_text, w_num], dim=-1) * self.weight_a)
                    .sum(dim=1)
                    .unsqueeze(0)
                    .T
                )
            else:
                w_num = None
                g_num = torch.zeros(0, device=g_text.device)

            alpha = torch.cat([g_text, g_cat, g_num], dim=1)  # N by 3
            alpha = F.leaky_relu(alpha, 0.02)
            alpha = F.softmax(alpha, -1)
            stack_tensors = [
                tensor for tensor in [w_text, w_cat, w_num] if tensor is not None
            ]
            combined = torch.stack(stack_tensors, dim=1)  # N by 3 by final_out_dim
            outputs_w_attention = alpha[:, :, None] * combined
            combined_feats = outputs_w_attention.sum(dim=1)  # N by final_out_dim
        elif self.combine_feat_method == "gating_on_cat_and_num_feats_then_sum":
            # assumes shifting of features relative to text features and that text features are the most important
            if cat_feats.shape[1] != 0:
                if self.cat_feat_dim > self.text_out_dim:
                    cat_feats = self.cat_layer(cat_feats)
                g_cat = self.dropout_cat(
                    self.act_func(
                        self.g_cat_layer(torch.cat([text_feats, cat_feats], dim=1))
                    )
                )
                g_mult_cat = g_cat * self.h_cat_layer(cat_feats)
            else:
                g_mult_cat = 0

            if numerical_feats.shape[1] != 0:
                if self.numerical_feat_dim > self.text_out_dim:
                    numerical_feats = self.num_layer(numerical_feats)
                g_num = self.dropout_num(
                    self.act_func(
                        self.g_num_layer(
                            torch.cat([text_feats, numerical_feats], dim=1)
                        )
                    )
                )
                g_mult_num = g_num * self.h_num_layer(numerical_feats)
            else:
                g_mult_num = 0

            H = g_mult_cat + g_mult_num + self.h_bias
            norm = torch.norm(text_feats, dim=1) / torch.norm(H, dim=1)
            alpha = torch.clamp(norm * self.tabular_config.gating_beta, min=0, max=1)
            combined_feats = text_feats + alpha[:, None] * H

        return combined_feats

    def __reset_parameters(self):
        glorot(self.weight_a)
        if hasattr(self, "weight_cat"):
            glorot(self.weight_cat)
            zeros(self.bias_cat)
        if hasattr(self, "weight_num"):
            glorot(self.weight_num)
            zeros(self.bias_num)
        glorot(self.weight_transformer)
        zeros(self.bias_transformer)
