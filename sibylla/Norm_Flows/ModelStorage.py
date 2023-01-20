"""Model storage class to deal with loading and saving model parameters and configs"""

import abc
import os
import json
import pickle


class ModelStorage(abc.ABC):
    def get_model_path(config, version=-1):
        """ Create a folder to save the model in and return the path"""
        model_path = os.path.join('trained_models', config.model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if version < 0:
            files = os.listdir(model_path)
            files.sort()
            target_version = files[version]
        else:
            target_version = f'version_{version}'
        version_path = os.path.join(model_path, target_version)
        return version_path

    def make_model_path(config):
        """ Create a folder to save the model in and return the path"""
        model_path = os.path.join('trained_models', config.model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        files = os.listdir(model_path)
        files.sort()
        if files != []:
            last_version = files[-1]
            last_version = last_version[len('version_'):]

            this_version = int(last_version) + 1
        else:
            this_version = 0

        version_path = os.path.join(model_path, f'version_{this_version}')
        os.makedirs(version_path)
        return version_path

    def save_config(save_path, config):
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config.to_json_best_effort(), f)

    def save_checkpoint(save_path, step, params):
        # remove old ckpt
        for file in os.listdir(save_path):
            if file.startswith("model_"):
                os.remove(os.path.join(save_path, file))

        # save new ckpt
        with open(os.path.join(save_path, f'model_{step}.pickle'), 'wb') as f:
            pickle.dump(params, f)

    def save_model(save_path, config, params):
        if config.eval.save_on_eval:
            # remove old ckpt
            for file in os.listdir(save_path):
                if file.startswith("model_"):
                    os.remove(os.path.join(save_path, file))
        with open(os.path.join(save_path, 'model.pickle'), 'wb') as f:
            pickle.dump(params, f)

    def load_model(save_path):
        with open(os.path.join(save_path, 'model.pickle'), 'rb') as f:
            params = pickle.load(f)
        return params
