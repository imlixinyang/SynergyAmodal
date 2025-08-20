import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from pytorch_lightning import LightningModule, Trainer

import warnings
warnings.filterwarnings("ignore")

from pytorch_lightning.callbacks import ModelCheckpoint

from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler

from utils import EMAModel, import_str


class RGBADecoderTrainer(LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)
        self.opt = opt    
        
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        
        self.latent_scale_fn = lambda x: x.latent_dist.sample() * 0.18215
        self.latent_unscale_fn = lambda x: x * (1 / 0.18215)
        
        self.vae = pipe.vae.eval().requires_grad_(False)
        self.vae.decoder = self.vae.decoder.train().requires_grad_(True)
        self.vae.decoder.conv_out = nn.Conv2d(self.vae.decoder.conv_out.weight.shape[1], 1, kernel_size=3, stride=1, padding=1)

        del pipe
        
        self.initialize_weights()

        self.model_ema = EMAModel(self.vae.decoder).eval()
        
    @torch.no_grad()
    def initialize_weights(self):        
        self.vae.decoder.conv_out.bias.data.fill_(0)
        self.vae.decoder.conv_out.weight.data.fill_(0)
        
    def configure_optimizers(self):
        params = [
            {'params': [param for param in list(self.vae.decoder.parameters()) if param.requires_grad],
             'lr': self.opt.training.learning_rate / self.opt.training.accumulate_grad_batches},
        ]
        
        optimizer = torch.optim.AdamW(params, weight_decay=self.opt.training.weight_decay, betas=self.opt.training.betas)
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 1000, 1))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step"}
        }
        
    def training_step(self, batch, _):

        self.model_ema.update_average(self.vae.decoder)
        
        image_output = batch['amodal_image']
        mask_output = batch['amodal_mask']
                             
        B = image_output.shape[0]
          
        with torch.no_grad():
            latents = self.latent_scale_fn(self.vae.encode(image_output * 2 - 1))

        mask_output_pred = self.vae.decode(self.latent_unscale_fn(latents.detach())).sample

        loss_bce = F.binary_cross_entropy_with_logits(mask_output_pred, mask_output, reduction="mean")

        mask_output_pred = mask_output_pred.flatten(1, 3).sigmoid()
        mask_output = mask_output.flatten(1, 3)
                
        numerator = 2 * (mask_output_pred * mask_output).sum(1)
        denominator = mask_output_pred.sum(1) + mask_output.sum(1)
        loss_dice = 1 - ((numerator + 0.1) / (denominator + 0.1)).mean()

        total_loss = loss_bce + loss_dice

        self.log(f'losses/loss_bce', loss_bce, sync_dist=True)
        self.log(f'losses/loss_dice', loss_dice, sync_dist=True)
        self.log(f'losses/total_loss', total_loss, sync_dist=True)
        
        return total_loss

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rgba_ours_pseudo.yaml", help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    try:
        num_nodes = int(os.environ["NUM_NODES"])
    except:
        os.environ["NUM_NODES"] = '1'
        num_nodes = 1
    
    print(num_nodes)

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    train_dataset = import_str(opt['dataset']['module'])(**opt['dataset']['args'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.training.batch_size, num_workers=opt.training.num_workers, shuffle=True, collate_fn=train_dataset.collate_fn)
        
    system = RGBADecoderTrainer(opt)

    trainer = Trainer(
            callbacks=[ModelCheckpoint(every_n_train_steps=1000)],
            default_root_dir=f"logs/{opt.experiment_name}",
            strategy='auto',
            max_steps=opt.training.max_steps,
            log_every_n_steps=1,
            accumulate_grad_batches=opt.training.accumulate_grad_batches,
            precision=opt.training.precision,
            accelerator='auto',
            devices=opt.training.devices,
            benchmark=True,
            gradient_clip_val=opt.training.gradient_clip_val,
            limit_val_batches=4,
            num_nodes=num_nodes,
        )
    
    trainer.fit(model=system, 
                train_dataloaders=train_loader, 
                ckpt_path=opt.training.resume_from_checkpoint)