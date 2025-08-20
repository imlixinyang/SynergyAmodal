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

from diffusers.training_utils import compute_snr

import random

from torchvision.utils import save_image

class InpaintingTrainer(LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)
        self.opt = opt    
        
        # TOFIX
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.requires_grad_(False).eval()
        
        self.latent_scale_fn = lambda x: x.latent_dist.sample() * 0.18215
        self.latent_unscale_fn = lambda x: x * (1 / 0.18215)
        
        self.vae = pipe.vae.requires_grad_(False).eval()

        self.unet = pipe.unet.requires_grad_(True).train()
        
        self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        
        del pipe
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = 0
        self.max_step = int(self.num_train_timesteps)   
        
        self.p_drop_condition = opt.training.p_drop_condition
        
        self.initialize_weights()
        
        self.model_ema = EMAModel(self.unet).eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                [''],
                padding="max_length",
                truncation_strategy='longest_first',
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_embedding = self.text_encoder(inputs.input_ids.to(next(self.text_encoder.parameters()).device))[0]
        
        self.register_buffer('text_embedding', text_embedding)
        
    @torch.no_grad()
    def initialize_weights(self):        
        weight = self.unet.conv_in.weight
        weight = F.pad(weight, (0, 0, 0, 0, 0, 4), value=0)
        self.unet.conv_in.weight = nn.Parameter(weight)
        
    def configure_optimizers(self):
        params = [
            {'params': list(self.unet.parameters()),
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
        
    def encode_text(self, text):
        input = self.tokenizer(
            text,
            padding="max_length",
            truncation_strategy='longest_first',
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_embedding = self.text_encoder(input.input_ids.to(self.device))[0]
        return text_embedding
    
    def training_step(self, batch, _):
        self.model_ema.update_average(self.unet)
        
        image = batch['image']
        modal_mask = batch['modal_mask']

        amodal_image = batch['amodal_image']
        amodal_mask = batch['amodal_mask']

        modal_image = image * modal_mask + 1 * (1 - modal_mask)
                                 
        B = image.shape[0]
          
        with torch.no_grad():
            
            batch['caption'] = [('' if random.random() < 0.5 else batch['caption'][i]) for i in range(B)]

            text_condition = self.encode_text(batch['caption'])

            all = torch.cat([image, modal_image, amodal_image], dim=0) * 2 - 1
            all_posteriors = self.vae.encode(all)
            all_latents = self.latent_scale_fn(all_posteriors)
            
            image_latent, modal_image_latent, amodal_image_latent = all_latents.split([B, B, B], dim=0)

            t = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=self.device)

            input_latents = amodal_image_latent

            # https://github.com/huggingface/diffusers/blob/78a78515d64736469742e5081337dbcf60482750/examples/text_to_image/train_text_to_image.py
            noise = torch.randn_like(input_latents) + 0.1 * torch.randn(B, input_latents.shape[1], 1, 1, device=self.device)
            
            input_latents_noisy = self.scheduler.add_noise(input_latents, noise, t)

            mask = torch.nn.functional.interpolate(
                1 - modal_mask, size=(64, 64)
            )
            
            image_condition = torch.cat([mask, modal_image_latent, image_latent], dim=1)
            
            image_condition = image_condition * (torch.rand(B, 1, 1, 1, device=self.device) > 0.1)

        # predict the noise residual
        noise_pred = self.unet(
                    torch.cat([input_latents_noisy, image_condition], dim=1),
                    t,
                    encoder_hidden_states=text_condition,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
        )[0]
        
        snr = compute_snr(self.scheduler, t)
        base_weight = (
                        torch.stack([snr, 5.0 * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr
                    )

        assert self.scheduler.config.prediction_type == "epsilon"
        # Epsilon and sample both use the same loss weights.
        mse_loss_weights = base_weight

        total_loss = (F.mse_loss(noise_pred, noise, reduction='none').mean([-1, -2, -3]) * mse_loss_weights).mean()
        
        self.log(f'losses/total_loss', total_loss, sync_dist=True)
        
        return total_loss
    
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
    def inference(self, batch, postprocess_latent=None, samples_per_condition=1, save_dir=None, **kwargs):
        """
        """
        image = batch['image']
        modal_mask = batch['modal_mask']

        modal_image = image * modal_mask + 1 * (1 - modal_mask)

        B = image.shape[0]
          
        with torch.no_grad():
            text_condition = self.encode_text(batch['caption'])

            all = torch.cat([image, modal_image], dim=0) * 2 - 1
            all_posteriors = self.vae.encode(all)
            all_latents = self.latent_scale_fn(all_posteriors)

            image_latent, modal_image_latent = all_latents.split([B, B], dim=0)     

            if save_dir is not None:
                save_image(((image_latent[0][:3] + 1) / 2).clamp(0, 1), os.path.join(save_dir, 'image_latent.png'))
                save_image(((modal_image_latent[0][:3] + 1) / 2).clamp(0, 1), os.path.join(save_dir, 'modal_image_latent.png'))
           
            mask = torch.nn.functional.interpolate(
                1 - modal_mask, size=(64, 64)
            )
            
            image_condition = torch.cat([mask, modal_image_latent, image_latent], dim=1)

            image_condition = image_condition.repeat(samples_per_condition, 1, 1, 1)
         
        # denoising
        noise = torch.randn_like(image_condition[:, :4])

        if save_dir is not None:
            save_image(((noise[0][:3] + 1) / 2).clamp(0, 1), os.path.join(save_dir, 'noise.png'))
        
        image_output_latents = self.inference_ddim(noise, image_condition, text_condition, **kwargs)

        if save_dir is not None:
            save_image(((image_output_latents[0][:3] + 1) / 2).clamp(0, 1), os.path.join(save_dir, 'image_output_latents.png'))

        image_output_latents = self.latent_unscale_fn(image_output_latents)
       
        image_output = (self.vae.decode(image_output_latents).sample * 0.5 + 0.5).clamp(0, 1)

        other_outputs = postprocess_latent(image_output_latents, batch) if postprocess_latent is not None else []
        
        return image_output, *other_outputs
    
    @torch.no_grad()
    def inference_ddim(self, noise, image_condition, text_condition, num_inference_steps=25, cfg_image=1, cfg_text=1, strength=1, initial_latent=None):
        B = noise.shape[0]
        assert self.num_train_timesteps % num_inference_steps == 0
        self.scheduler.config.steps_offset = self.num_train_timesteps // num_inference_steps - 1
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.scheduler.timesteps

        negative_text_condition = self.encode_text(['']).repeat(B, 1, 1)

        begin_index = int(len(timesteps) * (1 - strength))
        
        if initial_latent is None or begin_index == 0:
            latents_noisy = noise 
        else:
            latents_noisy = self.scheduler.add_noise(initial_latent.repeat(B, 1, 1, 1), noise, timesteps[begin_index][None].repeat(B))

        for i, t in enumerate(timesteps[begin_index:]):
            _t = t[None].repeat(B)
            
            if cfg_image == 1 and cfg_text == 1:
                noise_pred = self.unet(
                        torch.cat([latents_noisy, image_condition], dim=1),
                        timestep=t,
                        encoder_hidden_states=text_condition,
                    ).sample
            else:
                _latents_noisy = torch.cat([latents_noisy, latents_noisy, latents_noisy], dim=0)
                _text_condition = torch.cat([text_condition, negative_text_condition, negative_text_condition], dim=0)
                _image_condition = torch.cat([image_condition, image_condition, torch.zeros_like(image_condition)], dim=0)
                __t = torch.cat([_t, _t, _t], dim=0)
                
                noise_pred = self.unet(
                        torch.cat([_latents_noisy, _image_condition], dim=1),
                        timestep=__t,
                        encoder_hidden_states=_text_condition,
                    ).sample
                
                noise_pred, noise_pred1, noise_pred2 = noise_pred.chunk(3, 0)

                noise_pred = noise_pred2 + (noise_pred1 - noise_pred2) * cfg_image + (noise_pred - noise_pred1) * cfg_text
            
            latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy).prev_sample.to(self.device)

        return latents_noisy


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inpainter_ours_pseudo.yaml", help="path to the yaml config file")
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
        
    system = InpaintingTrainer(opt)

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