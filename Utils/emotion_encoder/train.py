"""
EmoStarGAN
Copyright (c) 2023-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import os.path as osp
import yaml
import shutil
import torch
import click
import warnings
warnings.simplefilter('ignore')

from munch import Munch
from Models.style_module import StyleEncoder
from Utils.emotion_encoder.dataset import build_dataloader
from optimizers import build_optimizer
from Utils.emotion_encoder.model import build_model
from Utils.emotion_encoder.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter


import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True

@click.command()
@click.option('-p', '--config_path', default='config.yml', type=str)

def main(config_path):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 1000)
    train_path = config.get('train_data', None)
    val_path = config.get('val_data', None)
    save_freq = config.get('save_freq', 20)
    stage = config.get('stage', 'star')
    fp16_run = config.get('fp16_run', False)
    domain = config.get('domain')

    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)
    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=4,
                                        device=device,
                                        domain=domain)
    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                      device=device,
                                      domain=domain)

    #ground_truth model
    GT_Model = StyleEncoder(64, 64, 5, 512)
    GT_Model_params = torch.load("emotion_style_encoder_pretrained_first_stage.pth",
                                 map_location='cpu')['model']['style_encoder']
    GT_Model.load_state_dict(GT_Model_params)
    GT_Model.to(device)

    # build model
    model, model_ema = build_model()

    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 2e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    _ = [model[key].to(device) for key in model]
    _ = [model_ema[key].to(device) for key in model_ema]
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                scheduler_params_dict=scheduler_params_dict)

    trainer = Trainer(args=Munch(config['loss_params']), model=model,
                      model_ema=model_ema,
                      optimizer=optimizer,
                      device=device,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      logger=logger,
                      gt_model = GT_Model,
                      fp16_run=fp16_run)
    eval_loss = float('inf')
    for _ in range(1, epochs+1):
        epoch = trainer.epochs
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        results = train_results.copy()
        results.update(eval_results)
        logger.info('--- epoch %d ---' % epoch)
        txt = ''
        for key, value in results.items():
            if isinstance(value, float):
                txt = txt + key + ':'+ format(value, ".4f") + '  '
                writer.add_scalar(key, value, epoch)
            else:
                for v in value:
                    writer.add_figure('eval_spec', v, epoch)
        logger.info(txt)
        if results["eval/coding_loss"] < eval_loss:
            text = "Evalulation losses " +  str(results["eval/coding_loss"]) + " is better than " + str(eval_loss) + ". Hence saving the model."
            logger.info(text)
            trainer.save_checkpoint(osp.join(log_dir, 'epoch_%05d.pth' % epoch))
            eval_loss = results["eval/coding_loss"]

    return 0


def get_data_path_list(train_path, val_path):
    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    return train_list, val_list

if __name__=="__main__":

    main()
