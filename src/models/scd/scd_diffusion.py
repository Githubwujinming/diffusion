from contextlib import contextmanager
import logging

from src.modules.ema import LitEma
from utils.metrics import weighted_BCE_logits
from .components import *
class SCDDiffNet(pl.LightningModule):
    def __init__(self,monitor=None, timesteps=[2],
                num_classes=7, mid_dim=128, diffusion_config=None,
                beta_schedule=None,embed_dim=1024,
                scheduler_config=None, ckpt_path=None, denoise_path=None) -> None:
        super().__init__()  
        self.save_hyperparameters(ignore=['loss_func', 'running_meters'])
        if monitor is not None:
            self.monitor = monitor 
        assert diffusion_config is not None
        self.decoder = SCDDecoder(embed_dim=embed_dim,num_classes=num_classes, mid_dim=mid_dim)
        self.loss_func = SCDLoss()
        self.train_running_meters = RunningMetrics(num_classes=num_classes)
        self.running_meters = RunningMetrics(num_classes=num_classes)
        self.scheduler_config = scheduler_config
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        self.init_denoise_from_ckpt(denoise_path, diffusion_config)
        
    
    def init_denoise_from_ckpt(self, path=None, diffusion_config=None):
        assert diffusion_config is not None
        self.encoder = instantiate_from_config(diffusion_config)
        self.encoder.set_new_noise_schedule(self.beta_schedule, self.device)

        if path is not None:
            self.encoder.load_state_dict(torch.load(
                path), strict=False)    
        # disable training of diffusion encoder
        self.encoder.eval()
        self.encoder.train = lambda *args: None
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        train_logger = logging.getLogger("train")
        train_logger.info("Loaded denoise model from {}".format(path))
        
    def encode(self, x1, x2):
        f_A = []
        f_B = []
        for t in self.timesteps:
            f_A.append(self.encoder.feats(x1, t)[0])
            f_B.append(self.encoder.feats(x2, t)[0])
        x1 = torch.cat(f_A, 1)
        x2 = torch.cat(f_B, 1)
        return [x1, x2]

    def decode(self, x, perturbation = None, o_l = None):
        return self.decoder(x, perturbation, o_l)
        
    def forward(self, x1, x2):
        x1, x2 = self.encode(x1, x2)
        return self.decode([x1, x2])
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    # 这里写反向传播过程
    def training_step(self, batch):
        x1 = batch['image_A'].clone()
        x2 = batch['image_B'].clone()
        bs = x1.size(0)
        target_A = batch['target_A'].long()
        target_B = batch['target_B'].long()
        x1, x2, change = self(x1, x2)
        target_bn = (target_B>0)
        loss = self.loss_func(x1, x2, change, target_bn.float(), target_A, target_B)
     
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, logger=True, batch_size=bs)
        
        sup_pred = self.get_prediction((x1, x2, change), domain='sup')
        self.train_running_meters.update(target_A.detach().cpu().numpy(), sup_pred['sup_pred_A'].detach().cpu().numpy())
        self.train_running_meters.update(target_B.detach().cpu().numpy(), sup_pred['sup_pred_B'].detach().cpu().numpy())
        return loss
    
    def on_train_epoch_end(self) -> None:
        # log metric for sup/unsup data
        sup_score = self.train_running_meters.get_scores()

        for k, v in sup_score.items():
            self.log(f'train/sup_{k}', v, on_step=False, on_epoch=True, logger=True)
        
        # logging in file
        train_logger = logging.getLogger('train')
        message = '[Training SCD (epoch %d summary)]: Fscd=%.5f \n' %\
                      (self.current_epoch, sup_score['F_scd'])
        for k, v in sup_score.items():
            message += '{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        train_logger.info(message)
        # reset mertric for next epoch
        self.running_meters.reset()
    
    # 这里配置优化器
    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.decoder.parameters()),
                                  lr=self.learning_rate)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                    {
                        'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                        'interval': 'step',
                        'frequency': 1
                    }
                    ]
            return [opt], scheduler
        return [opt], []
        
    
    def get_prediction(self, preds, domain='sup'):
        x1, x2, change = preds
        change_mask = torch.sigmoid(change).detach()>0.5
        p1 = torch.argmax(x1, dim=1)
        p2 = torch.argmax(x2, dim=1)
        p1 = p1 * change_mask.squeeze().long()
        p2 = p2 * change_mask.squeeze().long()
        del change_mask, x1, x2
        return {
            f'{domain}_pred_A': p1, 
            f'{domain}_pred_B': p2,
            # f'{domain}_change': change_mask.long(),
            }
    
   # 这里写一个epoch后的验证过程
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x1 = batch['image_A']
        x2 = batch['image_B']
        target_A = batch['target_A']
        target_B = batch['target_B']
        x1, x2, change = self(x1, x2)
        preds = dict()
        preds.update(self.get_prediction((x1, x2, change), domain='val'))
        
        self.running_meters.update(target_A.detach().cpu().numpy(), preds['val_pred_A'].detach().cpu().numpy())
        self.running_meters.update(target_B.detach().cpu().numpy(), preds['val_pred_B'].detach().cpu().numpy())
    
        return preds  
    
    def on_fit_start(self) -> None:
        val_logger = logging.getLogger('val')
        self.running_meters.reset()
        val_logger.info('\n###### EVALUATION ######')
        
    @torch.no_grad()
    def on_validation_epoch_end(self):
        # log the validation metrics
        val_logger = logging.getLogger('val')
        scores = self.running_meters.get_scores()
        self.log('val/F_scd', scores['F_scd'], on_epoch=True,  logger=True)
        self.log('val/miou', scores['Mean_IoU'], on_epoch=True, logger=True)
        self.log('val/OA', scores['OA'], on_epoch=True,  logger=True)
        self.log('val/SeK', scores['Sek'], on_epoch=True, logger=True)
        
        message  = f'Validation Summary Epoch: [{self.current_epoch}]\n'
        for k, v in scores.items():
            message += '{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        val_logger.info(message)
        self.running_meters.reset()
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x1, x2 = batch['image_A'], batch['image_B']
        image_id = batch['image_id'][0]
        target_A, target_B = batch['target_A'], batch['target_B']
        x1, x2, change = self(x1, x2)
        preds = dict()
        preds.update(self.get_prediction((x1, x2, change), domain='test'))
        self.running_meters.update(target_A.detach().cpu().numpy(), preds['test_pred_A'].detach().cpu().numpy())
        self.running_meters.update(target_B.detach().cpu().numpy(), preds['test_pred_B'].detach().cpu().numpy())
           
        preds['test_pred_A'] = Index2Color(preds['test_pred_A'].detach().cpu().numpy().squeeze())
        preds['test_pred_B'] = Index2Color(preds['test_pred_B'].detach().cpu().numpy().squeeze())
        # save images
        save_dir = os.path.join(self.trainer.log_dir)
        save_dir = os.path.join(save_dir, 'test_results')
        
        os.mkdir(save_dir) if not os.path.exists(save_dir) else None
        pred_A_L_show = PIL.Image.fromarray(preds['test_pred_A'].astype(np.uint8)).convert('RGB')
        pred_B_L_show = PIL.Image.fromarray(preds['test_pred_B'].astype(np.uint8)).convert('RGB')
        pred_A_L_show.save(os.path.join(save_dir, image_id+'_pred_A_L_show.png'))
        pred_B_L_show.save(os.path.join(save_dir, image_id+'_pred_B_L_show.png'))
        
        return preds
    def on_test_end(self) -> None:
        scores = self.running_meters.get_scores()
        message = '=========Test: performance=========\n'
        for k, v in scores.items():
            message += '{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        val_logger = logging.getLogger('val')
        val_logger.info(message)
        self.running_meters.reset()
        
    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        x1 = batch['image_A'], x2 = batch['image_B']
        image_id = batch['image_id']
        x1, x2, change = self(x1, x2)
        preds = {'image_id':image_id}
        preds.update(self.model.get_prediction((x1, x2, change), domain='test'))
        
        preds['test_pred_A'] = TensorIndex2Color(preds['test_pred_A']).detach().cpu().numpy().squeeze()
        preds['test_pred_B'] = TensorIndex2Color(preds['test_pred_B']).detach().cpu().numpy().squeeze()
        
        return preds
    
    @torch.no_grad()
    def log_images(self, batch, outputs, split='val', **kwargs):
        log = dict()
        A = batch['image_A']
        B = batch['image_B']
        denormalize = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        A = denormalize(A)# b,c,h,w
        B = denormalize(B)
        log['imageA'] = A
        log['imageB'] = B
        A_L = batch['target_A']# b,h,w,c
        B_L = batch['target_B']
        A_L = TensorIndex2Color(A_L)
        B_L = TensorIndex2Color(B_L)    
        log['target_A'] = A_L.permute(0,3,1,2)
        log['target_B'] = B_L.permute(0,3,1,2)
        log['pred_A'] = outputs[f'{split}_pred_A'].unsqueeze(1)
        log['pred_B'] = outputs[f'{split}_pred_B'].unsqueeze(1)
        return log

        