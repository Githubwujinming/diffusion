from main import *
import wandb
import pytz
def main(ks, stride, epochs):
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%m-%d_%H-%M')

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    # 
    sys.path.append(os.getcwd())
    # 解析命令给定的参数，保存在opt变量中，后面的cfg是解析配置文件获取的参数，
    # 如果有重复的参数，以命令行参数为准，相同设置由opt覆盖
    parser = get_parser()
    # 将Trainer可用的参数添加到parser中，例如gpu
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    # 获取继续训练的checkpoint
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            # 最新的checkpoint路径
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        # 设置继续训练的checkpoint的路径，后面会保存至trainer_config中，最后用于创建trainer
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = opt.name + "_"
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = cfg_name + "_"
        else:
            name = ""
        nowname = name + opt.postfix + now
        logdir = os.path.join(opt.logdir, nowname)
    # 设置目录
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    
    #setting logging
    os.makedirs(os.path.join(logdir, 'logs'), exist_ok=True)
    setup_logger('train', os.path.join(logdir, 'logs'), 'train', level=logging.INFO) 
    setup_logger('val', os.path.join(logdir, 'logs'), 'val', level=logging.INFO)
    train_logger = logging.getLogger('train')
    train_logger.info(f'================Training started in dir {logdir}======================')
    seed_everything(opt.seed)

    try:
        # init and save configs
        # 由omegaconf读取多个配置文件，可以将设置分散到多个配置文件中，如data.yaml, model.yaml, pl.yaml
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        # 对于未知的参数，使用omegaconf的from_dotlist方法解析
        cli = OmegaConf.from_dotlist(unknown)
        # 将已知的参数和未知的参数合并
        config = OmegaConf.merge(*configs, cli)
        # 获取lightning的配置，如果没有则创建一个
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        
        # default to ddp
        # trainer_config["accelerator"] = "ddp"
        # 对于trainer的参数，如果在命令行中指定了，则使用命令行中的参数，否则使用配置文件中的参数
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        # 如果没有指定gpu，则使用cpu noet: pl设置gpu的api已经改变了
        if not 'gpus' in trainer_config:
            # del trainer_config["accelerator"]
            cpu = True
        else:
            # 如果指定了gpu，则使用gpu
            gpuinfo = trainer_config["gpus"]
            train_logger.info(f"Running on GPUs {gpuinfo} \n")
            cpu = False
        # 将trainer的参数转换为命名空间,里面的属性可以通过.来访问
        trainer_opt = argparse.Namespace(**trainer_config)
        # trainer_config['max_epochs'] = epochs
        # 将设置好的trainer参数添加到lightning_config中
        lightning_config.trainer = trainer_config

        # data 获取数据集加载器
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        # 如果没有则下载
        # data.prepare_data()
        # 加载数据集
        data.setup()
        
        print("#### Data Setup #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # 计算每个epoch的迭代次数,在计算无监督部分的损失权重时需要用到
        if hasattr(config.data.params, "train_unsupervised"):
            iters_per_epoch = len(data.train_dataloader())
            num_epochs = lightning_config.trainer.max_epochs
            config.model.params.update({"iters_per_epoch": iters_per_epoch,
                                        "num_epochs": num_epochs})

        config.model.params.decoder_config.params.neck_config.params.update(
            {"ks": ks,
             "stride": stride}
        )
        
        # model 从配置文件中获取模型
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs 日志设置
        default_logger_cfgs = {
            "wandb": { # wandb日志
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "project": opt.project,
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                }
            },
        }
        # testtube 是pl内的日志工具, testtube在后面的版本取消了
        # 使用的是testtube日志，如果用wandb日志，则将default_logger_cfgs["testtube"]改为default_logger_cfgs["wandb"]
        default_logger_cfg = default_logger_cfgs[opt.g__logger_util]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        # 将日志设置添加到trainer_kwargs中
        # instantiate_from_config方法是根据配置文件中的target和params来实例化对象
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        # modelckpt_cfg是用来设置模型保存callback的参数,以epoch为单位保存模型
        default_modelckpt_cfg = {
            # ModelCheckpoint 的调用路径
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            # 传入ModelCheckpoint的参数
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:03}",# 模型保存的文件名
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"): # 最好模型依据的指标
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            # default_modelckpt_cfg["params"]["mode"] = 'max'
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 5# 保存最好的3个模型

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        # 将上面设置modelcheckpoint的参数合并到modelckpt_cfg中
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory 回调函数配置
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()
        # 设置多少个train_step保存一次模型, 
        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         "mode": 'max',
                         'save_top_k': -1,
                         'every_n_train_steps': 1000,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)
        # 更新callbacks_cfg
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        
        # 将callbacks_cfg中的参数添加到trainer_kwargs中
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        # 将设置的参数传入trainer中
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        
        # configure learning rate
        bs, base_lr = 8, config.model.base_learning_rate
        # gpu数量
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus)
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal
        # 信号处理，调用相应的函数
        signal.signal(signal.SIGUSR1, melk)# 如果有信号1，保存模型
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                # 训练模型
                trainer.fit(model, data)
            except Exception:
                # 如果出现异常，保存模型
                melk()
                raise
        else:
            # 指定某个模型进行测试
            if opt.resume:
                opt.no_test = True
                trainer.test(model, data, ckpt_path=ckpt)
        # 使用monitor指标最好的测试
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data, ckpt_path='best')
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            train_logger.info(trainer.profiler.summary())

if __name__ == "__main__":
    run = wandb.init()
    ks = wandb.config.ks
    stride = wandb.config.stride
    epochs = 10
    
    main(ks, stride, epochs)