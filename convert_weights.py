#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
from loguru import logger

import torch
import numpy as np
import pickle
import six

from pytorch.architectures.mymodel import MyModel
from pytorch.backbone.csp_darknet import CSPDarknet_small


def make_parser():
    parser = argparse.ArgumentParser("convert weights")

    # exp file
    parser.add_argument(
        "-b",
        "--backbone",
        default=None,
        type=str,
        help="backbone",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint")
    parser.add_argument("-oc", "--output_ckpt", default=None, type=str, help="output checkpoint")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    return parser



def copy(name, w, std):
    value2 = torch.Tensor(w)
    value = std[name]
    assert value.ndim == value2.ndim
    mul1 = np.prod(value.shape)
    mul2 = np.prod(value2.shape)
    assert mul1 == mul2
    value.copy_(value2)
    std[name] = value

def main(args):
    logger.info("Args: {}".format(args))
    # backbone名字
    backbone_name = args.backbone

    # 新增 backbone_name 时这里也要增加elif
    if backbone_name == 'CSPDarknet_small':
        backbone = CSPDarknet_small()
    else:
        raise NotImplementedError("backbone_name \'{}\' is not implemented.".format(backbone_name))

    model = MyModel(backbone)
    use_gpu = False
    if args.device == "gpu":
        model.cuda()
        use_gpu = True
    model.eval()
    model_std = model.state_dict()


    # 新增 backbone_name 时这里也要增加elif
    if backbone_name == 'CSPDarknet_small':
        with open(args.ckpt, 'rb') as f:
            state_dict = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
        # state_dict = fluid.io.load_program_state(args.ckpt)
        backbone_dic = {}
        for key, value in state_dict.items():
            if 'tracked' in key:
                continue
            else:
                backbone_dic['backbone.%s'%key] = value
        backbone_dic2 = {}
        others2 = {}
        for key, value in model_std.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic2[key] = value
            else:
                others2[key] = value
        for key in backbone_dic.keys():
            name2 = key
            w = backbone_dic[key]
            if 'StructuredToParameterName@@' in key:
                continue
            else:
                if '._mean' in key:
                    name2 = name2.replace('._mean', '.running_mean')
                if '._variance' in key:
                    name2 = name2.replace('._variance', '.running_var')
                if name2 in ['backbone.last_conv.weight', 'backbone.fc.weight', 'backbone.fc.bias']:
                    continue
                copy(name2, w, model_std)
    else:
        raise NotImplementedError("backbone_name \'{}\' is not implemented.".format(backbone_name))

    # save checkpoint.
    ckpt_state = {
        "start_epoch": 0,
        "model": model.state_dict(),
        "optimizer": None,
    }
    torch.save(ckpt_state, args.output_ckpt)
    logger.info("Done.")


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
