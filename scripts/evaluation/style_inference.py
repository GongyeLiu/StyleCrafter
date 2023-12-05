import argparse, os, sys, glob
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict

import torch
import torchvision
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
## note: decord should be imported after torch
from decord import VideoReader, cpu
from PIL import Image
import json
from torchvision.transforms import transforms
from torchvision.utils import make_grid

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler, DDIMStyleSampler
from utils.utils import instantiate_from_config
from utils.save_video import tensor_to_mp4


def save_img(img, path, is_tensor=True):
    if is_tensor:
        img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)

def get_filelist(data_dir, ext='*'):
    file_list = glob.glob(os.path.join(data_dir, '*.%s'%ext))
    file_list.sort()
    return file_list

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
    else:       
        # deepspeed
        state_dict = OrderedDict()
        for key in state_dict['module'].keys():
            state_dict[key[16:]]=state_dict['module'][key]

    model.load_state_dict(state_dict, strict=False)
    print('>>> model checkpoint loaded.')
    return model

def load_data_from_json(data_dir, filename=None, DISABLE_MULTI_REF=False):
    # load data from json file
    if filename is not None:
        json_file = os.path.join(data_dir, filename)
        with open(json_file, 'r') as f:
            data = json.load(f)
    else:
        json_file = get_filelist(data_dir, 'json')
        assert len(json_file) > 0, "Error: found NO prompt file!"
        default_idx = 0
        default_idx = min(default_idx, len(json_file)-1)
        if len(json_file) > 1:
            print(f"Warning: multiple prompt files exist. The one {os.path.split(json_file[default_idx])[1]} is used.")
        ## only use the first one (sorted by name) if multiple exist
        with open(json_file[default_idx], 'r') as f:
            data = json.load(f)

    n_samples = len(data)
    data_list = []

    style_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(512),
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x * 2. - 1.),
    ])

    for idx in range(n_samples):
        prompt = data[idx]['prompt']

        # load style image
        if data[idx]['style_path'] is not None:
            style_path = data[idx]['style_path']
            if isinstance(style_path, list) and not DISABLE_MULTI_REF:
                style_imgs = []
                for path in style_path:
                    style_img = Image.open(os.path.join(data_dir, path)).convert('RGB')
                    style_img_tensor = style_transforms(style_img)
                    style_imgs.append(style_img_tensor)
                style_img_tensor = torch.stack(style_imgs, dim=0)
            elif isinstance(style_path, list) and DISABLE_MULTI_REF:
                rand_idx = np.random.randint(0, len(style_path))
                style_img = Image.open(os.path.join(data_dir, style_path[rand_idx])).convert('RGB')
                style_img_tensor = style_transforms(style_img)
                print(f"Warning: multiple style images exist. The one {style_path[rand_idx]} is used.")
            else:
                style_img = Image.open(os.path.join(data_dir, style_path)).convert('RGB')
                style_img_tensor = style_transforms(style_img)
        else:
            raise ValueError("Error: style image path is None!")
            
        data_list.append({
            'prompt': prompt,
            'style': style_img_tensor
        })

    return data_list

def save_results(prompt, samples, filename, sample_dir, prompt_dir, fps=10, out_type='video'):
    ## save prompt
    prompt = prompt[0] if isinstance(prompt, list) else prompt
    path = os.path.join(prompt_dir, "%s.txt"%filename)
    with open(path, 'w') as f:
        f.write(f'{prompt}')
        f.close()

    ## save video
    if out_type == 'image':
        n = samples.shape[0]
        output = make_grid(samples, nrow=n, normalize=True, range=(-1, 1))
        output_img = Image.fromarray(output.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
        output_img.save(os.path.join(sample_dir, "%s.jpg"%filename))
    elif out_type == 'video':
        ## save video
        # b,c,t,h,w
        video = samples.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        path = os.path.join(sample_dir, "%s.mp4"%filename)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})
    else:
        raise ValueError("Error: output type should be image or video!")

def style_guided_synthesis(model, prompts, style, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, unconditional_guidance_scale_style=None, **kwargs):
    ddim_sampler = DDIMSampler(model) if unconditional_guidance_scale_style is None else DDIMStyleSampler(model) 

    batch_size = noise_shape[0]
    ## get condition embeddings (support single prompt only)
    if isinstance(prompts, str):
        prompts = [prompts]
    cond = model.get_learned_conditioning(prompts)
    # cond = repeat(cond, 'b n c -> (b f) n c', f=16)
    if unconditional_guidance_scale != 1.0:
        prompts = batch_size * [""]
        uc = model.get_learned_conditioning(prompts)
        # uc = repeat(uc, 'b n c -> (b f) n c', f=16)
    else:
        uc = None
    
    if len(style.shape) == 4:
        style_cond = model.get_batch_style(style)
        append_to_context = model.adapter(style_cond)
    else:
        bs, n, c, h, w = style.shape
        style = rearrange(style, "b n c h w -> (b n) c h w")
        style_cond = model.get_batch_style(style)
        style_cond = rearrange(style_cond, "(b n) l c -> b (n l ) c", b=bs)
        append_to_context = model.adapter(style_cond)
    # append_to_context = repeat(append_to_context, 'b n c -> (b f) n c', f=16)

    if hasattr(model.adapter, "scale_predictor"):
        scale_scalar = model.adapter.scale_predictor(torch.concat([append_to_context, cond], dim=1))
    else:
        scale_scalar = None

    batch_variants = []

    for _ in range(n_samples):
        if ddim_sampler is not None:
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_guidance_scale_style=unconditional_guidance_scale_style,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            append_to_context=append_to_context,
                                            scale_scalar=scale_scalar,
                                            **kwargs
                                            )    
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def run_inference(args, gpu_num, gpu_no):
    ## model config
    config = OmegaConf.load(args.base)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['adapter_config']['params']['scale'] = args.style_weight
    print(f"Set adapter scale to {args.style_weight:.2f}")
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"

    model = load_model_checkpoint(model, args.ckpt_path)
    model.load_pretrained_adapter(args.adapter_ckpt)
    if args.out_type == 'video' and args.temporal_ckpt is not None:
        model.load_pretrained_temporal(args.temporal_ckpt)
    model.eval()

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.channels
    frames = model.temporal_length if args.out_type == 'video' else 1
    noise_shape = [args.bs, channels, frames, h, w]

    sample_dir = os.path.join(args.savedir, "samples")
    prompt_dir = os.path.join(args.savedir, "prompts")
    style_dir = os.path.join(args.savedir, "style")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(style_dir, exist_ok=True)
    
    ## prompt file setting
    assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
    data_list = load_data_from_json(args.prompt_dir, args.filename, args.disable_multi_ref)
    num_samples = len(data_list)
    samples_split = num_samples // gpu_num
    print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
    #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    data_list_rank = [data_list[i] for i in indices]

    start = time.time() 
    for idx, indice in tqdm(enumerate(range(0, len(data_list_rank), args.bs)), desc='Sample Batch'):
        prompts = [batch_data['prompt'] for batch_data in data_list_rank[indice:indice+args.bs]]
        styles = [batch_data['style'] for batch_data in data_list_rank[indice:indice+args.bs]]

        if isinstance(styles, list):
            styles = torch.stack(styles, dim=0).to("cuda")
        else:
            styles = styles.unsqueeze(0).to("cuda")
        

        # if os.path.exists(os.path.join(args.savedir, 'style/{:04d}_style_randk{:d}.png'.format(idx + 1, gpu_no))):
        #     continue
        with torch.cuda.amp.autocast(dtype=torch.float32):
            batch_samples = style_guided_synthesis(model, prompts, styles, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                                args.unconditional_guidance_scale, args.unconditional_guidance_scale_style)
            if args.out_type == 'image':
                batch_samples = batch_samples[:, :, :, 0, :, :]
        
        if len(styles.shape) == 4:
            for nn in range(styles.shape[0]):
                filename = "%04d"%(idx*args.bs+nn + gpu_no * samples_split)
                save_img(styles[nn], os.path.join(style_dir, f'{filename}.png'))
        else:
            for nn in range(styles.shape[0]):
                filename = "%04d"%(idx*args.bs+nn + gpu_no * samples_split)
                for i in range(styles.shape[1]):
                    save_img(styles[nn, i], os.path.join(style_dir, f'{filename}_{i:02d}.png'))
        
        ## save each example individually
        for nn, samples in enumerate(batch_samples):
            ## samples : [n_samples,c,t,h,w]
            prompt = prompts[nn]
            filename = "%04d"%(idx*args.bs+nn + gpu_no * samples_split)
            for i in range(args.n_samples):
                save_results(prompt, samples[i:i+1], f"{filename}_{i}", sample_dir, prompt_dir, fps=10, out_type=args.out_type)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--adapter_ckpt", type=str, default=None, help="adapter checkpoint path")
    parser.add_argument("--temporal_ckpt", type=str, default=None, help="temporal checkpoint path")
    parser.add_argument("--base", type=str, help="config (yaml) path")
    parser.add_argument("--cond_type", default='style', type=str, help="conditon type: {style, depth, style_depth}")
    parser.add_argument("--out_type", default='video', type=str, help="output type: {image, video}")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--filename", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_style", type=float, default=None, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=0, help="seed for seed_everything")
    parser.add_argument("--style_weight", type=float, default=1.0)
    parser.add_argument("--disable_multi_ref", action='store_true', help="disable multiple style images")
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM cond-Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()

    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)