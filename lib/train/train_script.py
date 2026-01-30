import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.sutrack import build_sutrack
from lib.train.actors import SUTrack_Actor
from lib.utils.focal_loss import FocalLoss
# for import modules
import importlib


def run(settings):
    settings.description = 'Training script for Goku series'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg # generate cfg from lib.config
    config_module.update_config_from_file(settings.cfg_file) #update cfg from experiments
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_type = getattr(cfg.DATA, "LOADER", "tracking")
    if loader_type == "tracking":
        loader_train = build_dataloaders(cfg, settings)
    else:
        raise ValueError("illegal DATA LOADER")


    # Create network
    if settings.script_name == "sutrack":
        net = build_sutrack(cfg)
    else:
        raise ValueError("illegal script name")

        # =============================================================
        # ======================== 新增加载逻辑 ========================
        # 这里的 getattr 保证了如果 YAML 没写这个参数，程序也不会报错
    finetune_path = getattr(cfg.TRAIN, "FINETUNE_CHECKPOINT", "")
    if finetune_path != "" and settings.local_rank in [-1, 0]:
        if os.path.exists(finetune_path):
            print(f"==> 正在执行微调模式：从自定义路径加载全量权重: {finetune_path}")
            # 使用 cpu 加载以节省显存，随后 net.cuda() 会处理移动
            checkpoint = torch.load(finetune_path, map_location='cpu')
            # 自动处理权重包裹层级
            if 'net' in checkpoint:
                state_dict = checkpoint['net']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # strict=False 很重要，它能跳过不匹配的层（如不同数据集的分类头）
            missing, unexpected = net.load_state_dict(state_dict, strict=False)
            print(f"==> 加载成功！缺失键值数: {len(missing)}, 冗余键值数: {len(unexpected)}")
        else:
            print(f"==> 错误：找不到 FINETUNE_CHECKPOINT 文件: {finetune_path}")
    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, broadcast_buffers=False, device_ids=[settings.local_rank], find_unused_parameters=True) # modify the find_unused_parameters to False to skip a runtime error of twice variable ready
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    # Loss functions and Actors
    if  settings.script_name == "sutrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
