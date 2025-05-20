# 开发时间 2024/7/15 17:46
# 开发人员:牧良逢
import os
import copy
import pytorch_lightning as pl
from config import ex
from modal import DoubleEncoder
from data import F30kDataModule, MscocoDataModule, IAPRTC12DataModule, RSICDDataModule, ECommerceDataModule
import torch
def create_datamodule(_config, dataset_name):
    if 'f30k' in dataset_name:
        return F30kDataModule(_config)
    elif 'iaprtc12' in dataset_name:
        return IAPRTC12DataModule(_config)
    elif 'rsicd' in dataset_name:
        return RSICDDataModule(_config)
    elif 'ec' in dataset_name:
        return ECommerceDataModule(_config)
    else:
        return MscocoDataModule(_config)

ex.add_config({
    'train_type': 'VPT',
    #Prompt
    'use_prompt':True,
    'embed_dim':768,
    "prompt_pool": True,
    "pool_size": 10,
    "prompt_length": 5,
    "top_k":5,
    'prompt_init': 'uniform',
    'prompt_key': True,
    'prompt_key_init': 'uniform',
    'batchwise_prompt': False,
    'embedding_key': 'cls',
})

@ex.automain
def main(_config):
    print(_config)
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"], workers=True)
    model = DoubleEncoder(_config)
    model.cuda()
    #ckpt = torch.load(_config['checkpoint'])
    #model.load_state_dict(ckpt['state_dict'])
    #model.load_state_dict(ckpt)
    if _config['test_only']:
             ckpt = torch.load(_config['checkpoint'])
             model.load_state_dict(ckpt['state_dict'])
            # model=model #ZERO-SHOT

    os.makedirs(_config["log_dir"], exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        dirpath=f'PATH TO SAVE CHECKPOINT',
        #monitor="total_loss",
        #mode="min",
        save_last=True,
        filename='epoch={epoch}'
    )
    callbacks = [checkpoint_callback]
    #datasets_list = ['coco','f30k','iaprtc12','ec', 'rsicd'] 
    datasets_list = ['f30k','coco','rsicd','ec','iaprtc12']

    for idx,dataset_name in enumerate(datasets_list):
        print(f'当前数据集为{dataset_name}')
        print(f'当前的方法是{_config["train_type"]}')
        _config['datasets'] = dataset_name
        _config['task_ids']=idx
        print(f'Current task id is_{_config["task_ids"]}')
        dm = create_datamodule(_config, dataset_name)
        exp_name = f'finetune_irtr_{dataset_name}'
        _config['exp_name'] = exp_name
        print(_config['exp_name'])
        model.set_current_dataset(dataset_name)  # 设置当前数据集，用于模型调用相关的prompt
        logger = pl.loggers.TensorBoardLogger(
            _config["log_dir"],
            name=f'{exp_name}_seed{_config["seed"]}_fronbm_{_config["load_path"].split("/")[-1][:-5]}',
        )
        trainer = pl.Trainer(
            gpus=_config["num_gpus"],
            #gpus=[1],
            distributed_backend='ddp' if _config["num_gpus"] > 1 else None,
            precision=_config["precision"],
            benchmark=True,
            deterministic=True,
            max_epochs=_config["max_epoch"],
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=10,
            flush_logs_every_n_steps=10,
            weights_summary="top",
            val_check_interval=_config["val_check_interval"],
            #gradient_clip_val=10,
            gradient_clip_algorithm='norm'

        )
        if not _config["test_only"]:
            trainer.fit(model, datamodule=dm)
            torch.save(model.state_dict(), f'{_config["log_dir"]}/{dataset_name}_model.ckpt')
        else:
            trainer.test(model, datamodule=dm)
