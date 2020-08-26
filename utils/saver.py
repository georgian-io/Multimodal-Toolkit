from collections import defaultdict
import datetime
from os.path import join, getctime
import shutil
import pytz

import glob
from joblib import dump
import ntpath
import numpy as np
import pandas as pd
import pickle
from pprint import pprint
from tensorboardX import SummaryWriter
import torch

from utils.util import get_root_path, create_dir_if_not_exists, get_model_info_as_str


class Saver(object):
    def __init__(self, args, comet_exp_obj=None):
        """Object to handle all saving of models, training progress, and results
        to the local computer

        Args:
            comet_exp_obj: comet_ml.Experiment object if exists
        """
        model_str = self.get_model_str(args)
        experiment_dir_name = f'{model_str}_TVSplit{int(args.train_val_split * 100)}_{args.random_seed}_{get_current_ts()}'

        self.outer_log_dir = join(get_root_path(), 'logs', experiment_dir_name)
        create_dir_if_not_exists(self.outer_log_dir)
        if comet_exp_obj is not None:
            comet_exp_obj.log_parameter('log_dir', self.outer_log_dir)
        self.train_log_file = None
        self.log_training(f'Saving to {self.outer_log_dir}')
        with open(join(self.outer_log_dir, 'model_info.txt'), 'w') as f:
            f.write(get_model_info_as_str(args, self.outer_log_dir))
        self.cur_log_dir = None
        self.tb_cur_log_writer = None
        self.bert_model = True
        self.cur_run_results = None

    def _open(self, f):
        return open(join(self.outer_log_dir, f), 'w')

    def log_training(self, s):
        print(s)
        if self.train_log_file is None:
            self.train_log_file = self._open('train_log.txt')
        self.train_log_file.write(f'{s}\n')

    def update_cur_log_dir(self, fold_num, run_num):
        if self.tb_cur_log_writer is not None:
            self.tb_cur_log_writer.close()
        self.cur_log_dir = join(self.outer_log_dir,
                                f'fold_{fold_num}_run_{run_num}')
        create_dir_if_not_exists(self.cur_log_dir)
        self.tb_cur_log_writer = SummaryWriter(self.cur_log_dir)
        self.cur_run_results = defaultdict(list)

    def add_cur_run_results(self, result_dict, name='default'):
        assert self.cur_run_results is not None
        self.cur_run_results[name].append(result_dict)

    def save_cur_run_results(self):
        all_df = []
        for k, results in self.cur_run_results.items():
            res_df = pd.DataFrame(results)
            res_df = res_df.add_suffix('_' + k)
            all_df.append(res_df)
        fp = join(self.cur_log_dir, 'run_results.csv')
        all_df = pd.concat(all_df, axis=1)
        all_df = all_df[sorted(all_df.columns)]
        all_df.to_csv(fp)
        self.log_training(f'saved run results to {fp}')

    def log_tensorboard(self, tag, value, iter_num):
        self.tb_cur_log_writer.add_scalar(tag, value, iter_num)

    def log_tensorboard_from_dict(self, result_dict, iter_num):
        for key, val in result_dict.items():
            if type(val) == float or isinstance(val, np.floating):
                self.tb_cur_log_writer.add_scalar(key, val, iter_num)

    @staticmethod
    def get_model_str(args):
        """given the various flags return a string representing the name of our
        model and its basic configs"""
        li = []
        key_flags = [args.dataset, args.model, args.combine_feat_method]

        new_names = []
        for flag in key_flags:
             new_names.append(Saver.underscore_to_camelcase(flag))
        for f in new_names:
            li.append(str(f))
        return '_'.join(li)

    def save_to_results(self, args, res_dict, aggr=False):
        """saves the run results to a result.txt"""
        if not aggr:
            assert self.cur_log_dir is not None, 'need to update cur_log_dir'
            log_dir = self.cur_log_dir
        else:
            tb_writer = SummaryWriter(self.outer_log_dir)
            hparams = vars(args)
            hparams['log_dir'] = self.outer_log_dir
            self.__clean_hprams(hparams)
            tb_writer.add_hparams(hparam_dict=hparams, metric_dict=res_dict)

            log_dir = self.outer_log_dir
        with open(join(log_dir, f'results.txt'), 'a+') as f:
            assert type(res_dict) is dict
            pprint(res_dict, stream=f)

    def save_pred_scores(self, name, test_data, positive_pred_scores):
        fpath = join(self.cur_log_dir, f'{name}_positive_pred_scores.csv')
        tids = list(test_data.tid_map.keys())
        if type(test_data.labels) is torch.Tensor:
            labels = list(test_data.labels.cpu().detach().numpy())
        else:
            labels = list(test_data.labels)

        result_dict = {
            'tids': tids,
            'pred_scores': list(positive_pred_scores),
            'labels': labels
        }

        df = pd.DataFrame(result_dict)
        df.to_csv(fpath, index=False, index_label='tids')
        self.log_training(f'Saved pred scores to {fpath}')

    def save_trained_model(self, trained_model, epoch):
        """
        Saves a trained PyTorch model. Handles normal PyTorch models and
        HuggingFace Transformer models
        Args:
            trained_model: PyTorch model to be saved
            epoch (int): the training epoch of the current trained model
        """
        epoch_str = f'_epoch_{epoch}'
        folder = join(self.cur_log_dir, f'trained_model{epoch_str}')
        create_dir_if_not_exists(folder)
        if not hasattr(trained_model, 'save_pretrained'):
            save_path = join(folder, ntpath.basename(folder) + '.pt')
            torch.save(trained_model, save_path)
            self.bert_model = False
        else:
            save_path = folder
            trained_model.save_pretrained(folder)
            self.bert_model = True
        self.log_training('Trained model saved to {}'.format(save_path))

    def save_sklearn_model(self, clf):
        path = join(self.cur_log_dir, f'sklearn_model.joblib')
        dump(clf, path)
        self.log_training(f'Saved sklearn model to {path}')

    def save_pickle(self, obj, file_path):
        """ method to save any generic python object to the current run folder"""
        path = join(self.cur_log_dir, file_path)
        print('Saving to {}'.format(path))
        with open(path, 'wb') as pkl:
            pickle.dump(obj, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    def load_best_trained_model(self, model_class, device, delete_older_saved=True):
        """ Loads the best trained PyTorch model according to the time the model was saved
        Assumes the second latest model saved was the best model
        Args:
            model_class: class of the PyTorch model
            device: torch device
            delete_older_saved: whether to delete all the previous saved models
                that did not perform the best. Set to True to save disk space
        Returns:
            the loaded PyTorch Model
        """

        name = f'trained_model*'
        p = join(self.cur_log_dir, name)
        folders = glob.glob(p)
        f_sorted = sorted(folders, key=getctime, reverse=True)
        if delete_older_saved:
            for f_path in f_sorted[2:] + [f_sorted[0]]:
                try:
                    shutil.rmtree(f_path)
                except OSError as e:
                    print('Error: {}: {}'.format(f_path, e))

        best_trained_model_path = f_sorted[1]
        if not self.bert_model:
            load_path = join(best_trained_model_path, ntpath.basename(best_trained_model_path) + '.pt')
            model = torch.load(load_path)
        else:
            model = model_class.from_pretrained(best_trained_model_path)
        model.to(device)
        self.log_training('Loaded trained model from {}'.format(best_trained_model_path))
        return model

    def __del__(self):
        if self.tb_cur_log_writer is not None:
            self.tb_cur_log_writer.close()
        if self.train_log_file is not None:
            self.train_log_file.close()

    @staticmethod
    def __clean_hprams(hparam_dict):
        for key, val in hparam_dict.items():
            if type(val) == list:
                val = '__'.join(val)
                hparam_dict[key] = val
            elif val is None:
                hparam_dict[key] = 'None'

    @staticmethod
    def underscore_to_camelcase(word):
        return ''.join(x.capitalize() or '_' for x in word.split('_'))


def get_current_ts(zone='US/Pacific'):
    return datetime.datetime.now(pytz.timezone(zone)).strftime(
        '%Y-%m-%dT%H-%M-%S.%f')