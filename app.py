import os
import gradio as gr
import numpy as np
import torch
from addict import Dict
from PIL import Image
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from utils import move_to, crop_and_resize, put_back
import torch.nn.functional as F
from torchvision.utils import save_image

dilate_opt = lambda x, kernel_size=5: F.max_pool2d(x[None], kernel_size=kernel_size, stride=1, padding=kernel_size//2)[0]

save_tmp_dir = './tmp/app'

draw_points = False

class DataAppInference:
    Description="""
# Deocclusion Data App
## Remember to specify the workspace and user id first!
"""
    
    def __init__(self, configs=Dict()):
        
        self.configs = configs
        self.init_data()
        self.app = gr.Blocks()
        with self.app:
            self.init_app()
    
    def launch(self, only_layout=False, **kwargs):
        if not only_layout:
            self.init_models()
        self.app.launch(**kwargs)
    
    def queue(self, **kwargs):
        self.app.queue(**kwargs)

    def init_data(self) :
        self.data = Dict(
        )

    def init_models(self):
        from zim import zim_model_registry, ZimPredictor
        
        backbone = "vit_l"
        ckpt_p = "zim_ckpts/zim_vit_l_2092"

        self.zim = ZimPredictor(zim_model_registry[backbone](checkpoint=ckpt_p).to(self.configs.device).eval())

        args = Dict(
            config="configs/inpainter_ours_pseudo.yaml",
            ldm_ckpt_dir="ldm_ckpt_dir",
            vae_ckpt_dir="vae_ckpt_dir",
            device=self.configs.device
        )

        opt = OmegaConf.load(args.config)

        from trainer_deocclusion_pseudo import InpaintingTrainer

        system = InpaintingTrainer.load_from_checkpoint(os.path.join(args.ldm_ckpt_dir, os.listdir(args.ldm_ckpt_dir)[0]), map_location=args.device, strict=True, opt=opt).to(args.device)
    
        system.model_ema.load_ema_params(system.unet)
        del system.model_ema

        from trainer_deocclusion_pseudo_decoder import RGBADecoderTrainer
            
        mask_decoder = RGBADecoderTrainer.load_from_checkpoint(os.path.join(args.vae_ckpt_dir, os.listdir(args.vae_ckpt_dir)[0]), map_location=args.device, strict=True).to(args.device)

        mask_decoder.model_ema.load_ema_params(mask_decoder.vae.decoder)

        del mask_decoder.model_ema
        del mask_decoder.vae.encoder

        self.system = system
        self.mask_decoder = mask_decoder

    def init_app(self):

        data = gr.State(
            dict(
                prompt={
                    "prompt_type": [],
                    "input_point": [],
                    "input_label": [],
                    "multiimage_output": "True"}
            )
        )

        # with gr.Row():

        #     helps = gr.Markdown(value=Description) 

        with gr.Row():

            with gr.Column(scale=1.0):

                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(type="pil", interactive=True, label='Image', show_label=True)

                    with gr.Column(scale=1):
                        image_output = gr.Image(type="pil", interactive=False, label='Mask', show_label=True)
                
                    with gr.Column(scale=1):
                        amodal_output = gr.Image(type="pil", interactive=False, label='Mask', show_label=True)

                with gr.Row():
                    with gr.Column(scale=1):
                        pass

                    with gr.Column(scale=1):
                        point_prompt = gr.Radio(
                                        choices=["Positive", "Negative"],
                                        value="Positive",
                                        label="Point Prompt",
                                        interactive=True)
                                
                        clear_click_button = gr.Button(value="Clear Clicks", interactive=True)

                    with gr.Column(scale=1):

                        caption = gr.Textbox(label="Caption", interactive=True)
                        local_noise_strength_value = gr.Slider(0, 1, value=1, interactive=True, label="local_noise_strength")
                        erode_kernel_size_value = gr.Slider(1, 5, value=1, step=1, interactive=True, label="erode_kernel_size")
                        cfg_image_value = gr.Slider(0, 10, value=1, interactive=True, label="cfg_image")
                        cfg_text_value = gr.Slider(0, 10, value=1, interactive=True, label="cfg_text")
                        deocclude_button = gr.Button(value="Deocclude", interactive=True)

        clear_click_button.click(
            self.clear_click_button_click,
            inputs=[data],
            outputs=[image_output, point_prompt, data],
        )

        image_output.select(
            self.image_output_select,
            inputs=[point_prompt, data],
            outputs=[image_output, data],
        )

        image_input.upload(
            self.image_input_upload,
            inputs=[image_input, data],
            outputs=[image_output, point_prompt, data]
        )

        deocclude_button.click(
            self.deocclude_button_click,
            inputs=[data, caption, cfg_image_value, cfg_text_value, local_noise_strength_value, erode_kernel_size_value],
            outputs=[amodal_output]
        )   

    def clear_click_button_click(self, data): 
        data['prompt'] = {
            "prompt_type": [],
            "input_point": [],
            "input_label": [],
            "multiimage_output": "True",
        }
        return data['image_input_np'], "Positive", data

    def image_input_upload(self, image_input, data):
        data["prompt"] = {
            "prompt_type": [],
            "input_point": [],
            "input_label": [],
            "multiimage_output": "True",
        }
        res = 1024
        width, height = image_input.size
        ratio = 1.0 * res / max(width, height)
        image_input = image_input.resize((int(width * ratio + 0.5), int(height * ratio + 0.5)), Image.Resampling.BILINEAR)
        print('Scaling input image to {}'.format(image_input.size))

        data['image_input'] = image_input
        data['image_input_np'] = np.array(image_input)

        self.zim.set_image(data['image_input_np'])

        return image_input, gr.update(value="Positive"), data

    def image_output_select(self, point_prompt, data, evt: gr.SelectData):
        data['prompt']['prompt_type'] += ["click"]
        data['prompt']['input_point'] += [evt.index]
        data['prompt']['input_label'] += [1 if point_prompt == 'Positive' else 0]

        mask, _, _ = self.zim.predict(
                        point_coords=np.array(data['prompt']['input_point']),
                        point_labels=np.array(data['prompt']['input_label']),
                        multimask_output=False)

        mask = (mask > 0.5).astype(float)[0][:, :, None]

        data['mask'] = mask
        
        mask_np = mask

        image_output_np = data['image_input_np'] * mask_np + 0.2 * data['image_input_np'] * (1 - mask_np)

        image_output = Image.fromarray(image_output_np.astype(np.int8), 'RGB')

        image_output_draw = ImageDraw.Draw(image_output)

        if draw_points:
            for prompt_type, input_point, input_label in zip(data['prompt']['prompt_type'], data['prompt']['input_point'], data['prompt']['input_label']):
                if prompt_type == "click":
                    x = int(input_point[0])
                    y = int(input_point[1])

                x = min(max(x, 4), image_output.size[0] - 4)
                y = min(max(y, 4), image_output.size[1] - 4)
                
                image_output_draw.ellipse((x-4, y-4, x+4, y+4), outline ='blue' if input_label == 1 else 'red', width=4, fill=True)

        return image_output, data

    def deocclude_button_click(self, data, caption, cfg_image_value, cfg_text_value, local_noise_strength_value, erode_kernel_size_value):
        @torch.no_grad()
        def postprocess_latent(latent, *args):
            return self.mask_decoder.vae.decode(latent).sample.sigmoid().round(),

        image, modal_mask = torch.from_numpy(data['image_input_np']).permute(2, 0, 1).float() / 255., 1 - dilate_opt(1 - torch.from_numpy(data['mask'])[:, :, 0].float()[None], erode_kernel_size_value)[0]

        # save_image(image, os.path.join(save_tmp_dir, '1_image.png'))
        # save_image(modal_mask, os.path.join(save_tmp_dir, '1_mask.png'))
        # save_image(image * modal_mask[None], os.path.join(save_tmp_dir, '1_modal.png'))

        text = caption

        image_size = 512
                    
        H, W = image.shape[1:]
                    
        resize_scale = image_size / max(H, W)

        all = torch.cat([image, modal_mask[None]], dim=0)
        all = F.interpolate(all[None], size=(int(H * resize_scale), int(W * resize_scale)), mode='bilinear', align_corners=False)[0]

        all = F.pad(all, (0, image_size - all.shape[-1], 0, image_size - all.shape[-2]), value=0)
                    
        image, modal_mask = all.split([3, 1], dim=0)
                    
        modal_mask = (modal_mask > 0.5).float()

        # save_image(image, os.path.join(save_tmp_dir, '2_image.png'))
        # save_image(modal_mask, os.path.join(save_tmp_dir, '2_mask.png'))
                    
        fbatch = {
                        'image': image[None],
                        'modal_mask': modal_mask[None],
                        'caption': [text],
                    }
                    
        fbatch = move_to(fbatch, self.configs.device)
                    
        amodal_image_output, amodal_mask_output = self.system.inference(fbatch, cfg_image=cfg_image_value, cfg_text=cfg_text_value, postprocess_latent=postprocess_latent)

        # save_image(amodal_image_output, os.path.join(save_tmp_dir, '3_image.png'))
        # save_image(amodal_mask_output, os.path.join(save_tmp_dir, '3_mask.png'))
                    
        amodal_rgba = torch.cat([amodal_image_output, amodal_mask_output], dim=1)[0].cpu()
                    
        amodal_rgba = amodal_rgba[:, :int(H * resize_scale), :int(W * resize_scale)]
                    
        amodal_rgba = F.interpolate(amodal_rgba[None], size=(H, W), mode='bilinear', align_corners=False)[0]
                    
        amodal_image_output, amodal_mask_output = amodal_rgba.split([3, 1], dim=0)
                    
        amodal_mask_output = (amodal_mask_output > 0.5).float()

        amodal_image_output = amodal_image_output * amodal_mask_output + 1 * (1 - amodal_mask_output)

        # save_image(amodal_image_output, os.path.join(save_tmp_dir, '4_image.png'))
        # save_image(amodal_mask_output, os.path.join(save_tmp_dir, '4_mask.png'))
                    
        # coarse-to-fine
        image, modal_mask = torch.from_numpy(data['image_input_np']).permute(2, 0, 1).float() / 255., 1 - dilate_opt(1 - torch.from_numpy(data['mask'])[:, :, 0].float()[None], erode_kernel_size_value)[0]
                    
        cropped_all, bbox = crop_and_resize(torch.cat([image, modal_mask[None], amodal_image_output], dim=0), amodal_mask_output[0], min_size=256, padding_value=0)
        cropped_image, cropped_modal_mask, cropped_amodal_image_output = cropped_all.split([3, 1, 3], dim=0)

        # save_image(cropped_image, os.path.join(save_tmp_dir, '5_image.png'))
        # save_image(cropped_modal_mask, os.path.join(save_tmp_dir, '5_mask.png'))
        # save_image(cropped_amodal_image_output, os.path.join(save_tmp_dir, '5_amodal_image.png'))
                    
        cropped_modal_mask = (cropped_modal_mask > 0.5).float()
                    
        batch = {
            'image': cropped_image[None],
            'modal_mask': cropped_modal_mask[None],
            'caption': [text],
        }
                        
        batch = move_to(batch, self.configs.device)
                        
        cropped_amodal_image_output, cropped_amodal_mask_output = self.system.inference(batch, cfg_image=cfg_image_value, cfg_text=cfg_text_value,postprocess_latent=postprocess_latent, 
                    strength=local_noise_strength_value, 
                    initial_latent=self.system.latent_scale_fn(self.system.vae.encode(cropped_amodal_image_output[None].contiguous().to(self.configs.device) * 2 - 1)),
                    save_dir=save_tmp_dir)
        
        # save_image(cropped_amodal_image_output, os.path.join(save_tmp_dir, '6_amodal_image.png'))
        # save_image(cropped_amodal_mask_output, os.path.join(save_tmp_dir, '6_amodal_mask.png'))
        
        cropped_amodal_rgba = torch.cat([cropped_amodal_image_output, cropped_amodal_mask_output], dim=1)[0].cpu()
                        
        amodal_rgba = put_back(cropped_amodal_rgba, H, W, bbox)
                    
        amodal_rgba[3] = (amodal_rgba[3] > 0.5).float()

        # save_image(amodal_rgba[:3] * amodal_rgba[3][None] + 1 * (1 - amodal_rgba[3][None]), os.path.join(save_tmp_dir, '7_image.png'))
        # save_image(amodal_rgba[3], os.path.join(save_tmp_dir, '7_mask.png'))

        return amodal_rgba.float().permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--device', type=str, default='cuda:0')
    args.add_argument('--port', type=int, default=7860)
    args = args.parse_args()


    app = DataAppInference(configs=Dict(workspace='default', device=args.device))
    app.launch(share=False, only_layout=False, 
               server_name="0.0.0.0", server_port=args.port, ssl_verify=False,
               ssl_certfile="cert.pem",
               ssl_keyfile="key.pem")
