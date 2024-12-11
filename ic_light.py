import os
import math
import numpy as np
import torch
import safetensors.torch as sf
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from torch.hub import download_url_to_file
from enum import Enum

class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

class ICLight:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.models_dir = './models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 모델 초기화
        self._initialize_models()
        self._setup_unet()
        self._load_ic_light_weights()
        self._move_to_device()
        self._setup_attention()
        self._setup_schedulers()
        self._setup_pipelines()

    def _initialize_models(self):
        """모델 초기화"""
        sd15_name = 'stablediffusionapi/realistic-vision-v51'
        self.tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
        self.rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

    def _setup_unet(self):
        """UNet 설정"""
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(8, self.unet.conv_in.out_channels, 
                                        self.unet.conv_in.kernel_size,
                                        self.unet.conv_in.stride, 
                                        self.unet.conv_in.padding)
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            new_conv_in.bias = self.unet.conv_in.bias
            self.unet.conv_in = new_conv_in

        self.unet_original_forward = self.unet.forward
        self.unet.forward = self._hooked_unet_forward

    def _hooked_unet_forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        """UNet forward hook"""
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return self.unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

    def _load_ic_light_weights(self):
        """IC-Light 가중치 로드"""
        model_path = os.path.join(self.models_dir, 'iclight_sd15_fc.safetensors')
        if not os.path.exists(model_path):
            download_url_to_file(
                'https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors',
                model_path
            )

        sd_offset = sf.load_file(model_path)
        sd_origin = self.unet.state_dict()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        self.unet.load_state_dict(sd_merged, strict=True)

    def _move_to_device(self):
        """모델을 디바이스로 이동"""
        self.text_encoder = self.text_encoder.to(device=self.device, dtype=torch.float16)
        self.vae = self.vae.to(device=self.device, dtype=torch.bfloat16)
        self.unet = self.unet.to(device=self.device, dtype=torch.float16)
        self.rmbg = self.rmbg.to(device=self.device, dtype=torch.float32)

    def _setup_attention(self):
        """어텐션 프로세서 설정"""
        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae.set_attn_processor(AttnProcessor2_0())

    def _setup_schedulers(self):
        """스케줄러 설정"""
        self.dpm_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1
        )

    def _setup_pipelines(self):
        """파이프라인 설정"""
        self.t2i_pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.dpm_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None
        )

        self.i2i_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.dpm_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None
        )

    def process_relight(
        self,
        input_fg,
        prompt,
        image_width=512,
        image_height=768,
        num_samples=1,
        seed=12345,
        steps=25,
        a_prompt="best quality",
        n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
        cfg=2.0,
        highres_scale=1.5,
        highres_denoise=0.5,
        lowres_denoise=0.9,
        bg_source="Right Light"
    ):
        input_fg, matting = self.run_rmbg(input_fg)
        """이미지 처리 및 생성"""
        bg_source = BGSource(bg_source)

        # 배경 생성
        if bg_source == BGSource.NONE:
            pass
        elif bg_source == BGSource.LEFT:
            gradient = np.linspace(255, 0, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.RIGHT:
            gradient = np.linspace(0, 255, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.TOP:
            gradient = np.linspace(255, 0, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.BOTTOM:
            gradient = np.linspace(0, 255, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        else:
            raise ValueError('Wrong initial latent!')

        # 입력 이미지 전처리
        input_fg = Image.fromarray(input_fg).convert('RGB')
        
        # center crop 적용
        fg = resize_and_center_crop(np.array(input_fg), image_width, image_height)

        with torch.no_grad():
            # 생성기 설정
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # 조건 인코딩
            concat_conds = numpy2pytorch([fg]).to(device=self.vae.device, dtype=self.vae.dtype)
            concat_conds = self.vae.encode(concat_conds).latent_dist.mode() * self.vae.config.scaling_factor
            
            conds, unconds = encode_prompt_pair(
                self.tokenizer, 
                self.text_encoder,
                positive_prompt=prompt + ', ' + a_prompt,
                negative_prompt=n_prompt,
                device=self.device
            )

            # 이미지 생성
            if input_bg is None:
                latents = self.t2i_pipe(
                    prompt_embeds=conds,
                    negative_prompt_embeds=unconds,
                    width=image_width,
                    height=image_height,
                    num_inference_steps=steps,
                    num_images_per_prompt=num_samples,
                    generator=generator,
                    output_type='latent',
                    guidance_scale=cfg,
                    cross_attention_kwargs={'concat_conds': concat_conds},
                ).images.to(self.vae.dtype) / self.vae.config.scaling_factor
            else:
                bg = resize_and_center_crop(input_bg, image_width, image_height)
                bg_latent = numpy2pytorch([bg]).to(device=self.vae.device, dtype=self.vae.dtype)
                bg_latent = self.vae.encode(bg_latent).latent_dist.mode() * self.vae.config.scaling_factor
                bg_latent = bg_latent.repeat(num_samples, 1, 1, 1)
                
                latents = self.i2i_pipe(
                    image=bg_latent,
                    strength=lowres_denoise,
                    prompt_embeds=conds,
                    negative_prompt_embeds=unconds,
                    num_inference_steps=int(round(steps / lowres_denoise)),
                    num_images_per_prompt=num_samples,
                    generator=generator,
                    output_type='latent',
                    guidance_scale=cfg,
                    cross_attention_kwargs={'concat_conds': concat_conds},
                ).images.to(self.vae.dtype) / self.vae.config.scaling_factor

            # 고해상도 처리
            pixels = self.vae.decode(latents).sample
            pixels = pytorch2numpy(pixels)
            pixels = [resize_without_crop(
                image=p,
                target_width=int(round(image_width * highres_scale / 64.0) * 64),
                target_height=int(round(image_height * highres_scale / 64.0) * 64))
                for p in pixels]

            pixels = numpy2pytorch(pixels).to(device=self.vae.device, dtype=self.vae.dtype)
            latents = self.vae.encode(pixels).latent_dist.mode() * self.vae.config.scaling_factor
            latents = latents.to(device=self.unet.device, dtype=self.unet.dtype)

            # 최종 이미지 크기 조정
            image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8
            fg = resize_and_center_crop(input_fg, image_width, image_height)
            concat_conds = numpy2pytorch([fg]).to(device=self.vae.device, dtype=self.vae.dtype)
            concat_conds = self.vae.encode(concat_conds).latent_dist.mode() * self.vae.config.scaling_factor

            # 최종 이미지 생성
            latents = self.i2i_pipe(
                image=latents,
                strength=highres_denoise,
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                num_inference_steps=int(round(steps * highres_denoise)),
                num_images_per_prompt=num_samples,
                generator=generator,
                output_type='latent',
                guidance_scale=cfg,
                cross_attention_kwargs={'concat_conds': concat_conds},
            ).images.to(self.vae.dtype) / self.vae.config.scaling_factor

            # 최종 이미지 디코딩
            pixels = self.vae.decode(latents).sample
            outputs = pytorch2numpy(pixels)

            return fg, outputs

    def _setup_pipelines(self):
        """파이프라인 설정"""
        self.t2i_pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.dpm_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None
        )

        self.i2i_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.dpm_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None
        )

    @torch.inference_mode()
    def run_rmbg(self, img, sigma=0.0):
        """배경 제거"""
        H, W, C = img.shape
        assert C == 3
        k = (256.0 / float(H * W)) ** 0.5
        feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
        feed = numpy2pytorch([feed]).to(device=self.device, dtype=torch.float32)
        alpha = self.rmbg(feed)[0][0]
        alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
        alpha = alpha.movedim(1, -1)[0]
        alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
        result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
        return result.clip(0, 255).astype(np.uint8), alpha

def get_ic_light(device='cuda:0'):
    """IC-Light 인스턴스 생성"""
    return ICLight(device=device)

# 유틸리티 함수들
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)
        results.append(y)
    return results

def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    h = h.movedim(-1, 1)
    return h

def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

def encode_prompt_pair(tokenizer, text_encoder, positive_prompt, negative_prompt, device):
    """프롬프트 인코딩"""
    max_length = tokenizer.model_max_length
    
    def encode_prompt(prompt):
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)
        return text_encoder(tokens)[0]
    
    conds = encode_prompt(positive_prompt)
    unconds = encode_prompt(negative_prompt)
    
    return conds, unconds