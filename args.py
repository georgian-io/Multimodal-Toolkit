from dataclasses import dataclass, field
import json
import logging
from os.path import join
from typing import Optional, Tuple

import torch
from transformers.training_args import TrainingArguments, torch_required, cached_property

from utils.util import get_log_path

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )



@dataclass
class MultimodalDataTrainingArguments:
    """
    Arguments pertaining to how we combine tabular features
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_path: str = field(metadata={
                              'help': 'the path to the csv file containing the dataset'
                          })
    column_info_path: str = field(metadata={
        'help': 'the path to the json file detailing which columns are text, categorical, numerical, and the label'
    })

    categorical_encode_type: str = field(default='ohe',
                                         metadata={
                                             'help': 'sklearn encoder to use for categorical data',
                                             'choices': ['ohe', 'binary', 'label', 'none']
                                         })
    numerical_transformer_method: str = field(default='yeo_johnson',
                                              metadata={
                                                  'help': 'sklearn numerical transformer to preprocess numerical data',
                                                  'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']
                                              })
    task: str = field(default="classification",
                      metadata={
                          "help": "The downstream training task",
                          "choices": ["classification", "regression"]
                      })

    mlp_division: int = field(default=4,
                              metadata={
                                  'help': 'the ratio of the number of '
                                          'hidden dims in a current layer to the next MLP layer'
                              })
    combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat',
                                     metadata={
                                         'help': 'method to combine categorical and numerical features, '
                                                 'see README for all the method'
                                     })
    mlp_dropout: float = field(default=0.1,
                               metadata={
                                 'help': 'dropout ratio used for MLP layers'
                               })
    numerical_bn: bool = field(default=True,
                               metadata={
                                   'help': 'whether to use batchnorm on numerical features'
                               })
    use_simple_classifier: str = field(default=True,
                                       metadata={
                                           'help': 'whether to use single layer or MLP as final classifier'
                                       })
    mlp_act: str = field(default='relu',
                         metadata={
                             'help': 'the activation function to use for finetuning layers',
                             'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']
                         })
    gating_beta: float = field(default=0.2,
                               metadata={
                                   'help': "the beta hyperparameters used for gating tabular data "
                                           "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"
                               })

    def __post_init__(self):
        with open(self.column_info_path, 'r') as f:
            self.column_info = json.load(f)


@dataclass
class OurTrainingArguments(TrainingArguments):
    experiment_name: Optional[str] = field(
        default=None,
        metadata={'help': 'A name for the experiment'}
    )

    gpu_num: int = field(
        default=0,
        metadata={'help': 'The gpu number to train on'}
    )

    debug: bool = field(
        default=False,
        metadata={'help': 'Whether we are training in debug mode (smaller model)'}
    )

    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=True, metadata={"help": "Whether to run predictions on the test set."})

    evaluate_during_training: bool = field(
        default=True, metadata={"help": "Run evaluation during training at each logging step."},
    )

    max_token_length: Optional[int] = field(
        default=None,
        metadata={'help': 'The maximum token length'}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})

    def __post_init__(self):
        if self.output_dir is "":
            self.output_dir = join(get_log_path(),
                                   '_'.join(self.experiment_name.split(' ')))
        if self.debug:
            self.max_token_length = 16
            self.logging_steps = 5
            self.overwrite_output_dir = True


    @cached_property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device, n_gpu