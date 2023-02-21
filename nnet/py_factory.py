import os
import shutil

import torch
import importlib
import torch.nn as nn
from thop import profile, clever_format
from copy import deepcopy
from loguru import logger
from config import system_configs
from models.py_utils.data_parallel import DataParallel

torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, iteration, save, viz_split,
                xs, ys, **kwargs):
        preds, weights = self.model(*xs, **kwargs)
        loss  = self.loss(iteration,
                          save,
                          viz_split,
                          preds,
                          ys,
                          **kwargs)
        return loss

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, flag: bool = False, num_gpu: int = None):
        super(NetworkFactory, self).__init__()
        module_file = "models.{}".format(system_configs.snapshot_name)
        nnet_module = importlib.import_module(module_file)

        self.model = DummyModule(nnet_module.model(flag=flag))
        self.loss = nnet_module.loss()
        self.network = Network(self.model, self.loss)

        # logger.info("Images of one batch are split on multi-GPU with: {}".format(system_configs.chunk_sizes))
        self.network = DataParallel(self.network, chunk_sizes=system_configs.chunk_sizes, device_ids=list(range(num_gpu)))
        # self.network = DataParallel(self.network, chunk_sizes=system_configs.chunk_sizes)

        # True: Training / False: Testing
        self.flag    = flag

        # Count total parameters
        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params

        # Count MACs when input is 360 x 640 x 3
        # input_test = torch.randn(1, 3, 192, 960)#.cuda()
        # input_mask = torch.randn(1, 3, 192, 960)#.cuda()
        # macs, params, = profile(self.model, inputs=(input_test, input_mask), verbose=False)
        # macs, _ = clever_format([macs, params], "%.3f")
        # print('MACs: {}'.format(macs))

    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train(self,
              iteration,
              save,
              viz_split,
              xs,
              ys):
        xs = [x.cuda(non_blocking=True) for x in xs]#64,3,192,960 64,1,192,960
        ys = [y.cuda(non_blocking=True) for y in ys]

        self.optimizer.zero_grad()
        loss_kp = self.network(iteration,
                               save,
                               viz_split,
                               xs,
                               ys)

        loss      = loss_kp[0]
        loss_dict = loss_kp[1:]
        loss      = loss.mean()
        loss.backward()
        self.optimizer.step()

        return loss, loss_dict

    def validate(self,
                 iteration,
                 save,
                 viz_split,
                 xs,
                 ys):

        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]
            loss_kp = self.network(iteration,
                                   save,
                                   viz_split,
                                   xs,
                                   ys)
            loss      = loss_kp[0]
            loss_dict = loss_kp[1:]
            loss      = loss.mean()

            return loss, loss_dict

    def test(self, xs, **kwargs):
        with torch.no_grad():
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        logger.info("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def set_optimizer(self):

        if system_configs.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate,
                momentum=0.9, weight_decay=0.0001
            )
        elif system_configs.opt_algo == 'adamW':
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate,
                weight_decay=1e-4
            )
        else:
            raise ValueError("unknown optimizer")
    # def save_params(self, iteration, update_best_ckpt, model_name=""):
    #     save_model = self.model
    #     if len(model_name) > 0:
    #         cache_file = system_configs.snapshot_file.format(model_name)
    #     else:
    #         cache_file = system_configs.snapshot_file.format("latest")
    #     logger.info("saving model to {}".format(cache_file))
    #     ckpt_state = {
    #         "start_iter": iteration,
    #         "model": save_model.state_dict(),
    #         "optimizer": self.optimizer.state_dict(),
    #     }
    #     torch.save(ckpt_state, cache_file)
    #     if update_best_ckpt:
    #         best_filename = system_configs.snapshot_file.format("best")
    #         shutil.copyfile(cache_file, best_filename)

    def load_params(self, iteration, is_bbox_only=False):
        cache_file = system_configs.snapshot_file.format(iteration)

        with open(cache_file, "rb") as f:
            params = torch.load(f)
            model_dict = self.model.state_dict()
            '''
            for mk, mv in model_dict.items():
                print('=======',mk,mv.shape)
            for pk, pv in params.items():
                print('+++++',pk,pv.shape)
            #assert False
            '''
            
            if len(params) != len(model_dict):
                pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
            else:
                pretrained_dict = params
            model_dict.update(pretrained_dict)

            self.model.load_state_dict(model_dict)

    def save_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)

    def load_ckpt(self, ckpt):
        model_state_dict = self.model.state_dict()
        load_dict = {}
        # skip_dict = { 'module.specific_embed.layers.0.weight',
        #               'module.specific_embed.layers.0.bias',
        #               'module.specific_embed.layers.1.weight',
        #               'module.specific_embed.layers.1.bias',
        #               'module.specific_embed.layers.2.weight',
        #               'module.specific_embed.layers.2.bias'}
        skip_dict = {}
        for key_model, v in model_state_dict.items():
            if key_model not in ckpt:
                logger.warning(
                    "{} is not in the ckpt. Please double check and see if this is desired.".format(
                        key_model
                    )
                )
                continue
            if key_model in skip_dict:
                continue
            if key_model == 'module.specific_embed.layers.2.weight':
                v_ckpt = ckpt[key_model]#原先的输出通道数为8，现在的输出通道数为7需要丢掉最后一个通道的权重
            elif key_model == 'module.specific_embed.layers.2.bias':
                v_ckpt = ckpt[key_model]#同理
            else:
                v_ckpt = ckpt[key_model]
            #v_ckpt = ckpt[key_model]
            if v.shape != v_ckpt.shape:
                logger.warning(
                    "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                        key_model, v_ckpt.shape, key_model, v.shape
                    )
                )
                continue
            load_dict[key_model] = v_ckpt
        self.model.load_state_dict(load_dict, strict=False)

    def freeze_mode(self, freeze_dict):
        for (name, param) in self.model.named_parameters():
            if name.split(".")[1] in freeze_dict:
                param.requires_grad = False

    def resume_train(self, ckpt, resume):
        start_iter = 0
        if resume:
            logger.info("resume training")
            if ckpt is None:
                ckpt_file = system_configs.snapshot_file.format("latest")
            else:
                ckpt_file = ckpt
            ckpt = torch.load(ckpt_file)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_iter = ckpt["start_iter"]
            logger.info("loaded checkpoint '{}' (iter {})".format(ckpt_file, start_iter))  # noqa
        else:
            if ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt = torch.load(ckpt)
                self.load_ckpt(ckpt)
        return start_iter
