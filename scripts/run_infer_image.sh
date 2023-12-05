name="style_image_generation"
config="configs/inference_image_512_512.yaml"
ckpt="checkpoints/videocrafter_t2v_320_512/model.ckpt"
adapter_ckpt="checkpoints/stylecrafter/adapter_v1.pth"
prompt_dir="eval_data"
filename="eval_image_gen.json"
res_dir="output"
seed=123
n_samples=1


use_ddp=0
# set use_ddp=1 if you want to use multi GPU
# export CUDA_VISIBLE_DEVICES=2
if [ $use_ddp == 0 ]; then
python3 scripts/evaluation/style_inference.py \
--out_type 'image' \
--adapter_ckpt $adapter_ckpt \
--seed $seed \
--ckpt_path $ckpt \
--base $config \
--savedir $res_dir/$name \
--n_samples $n_samples \
--bs 1 --height 512 --width 512 \
--unconditional_guidance_scale 6.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--filename $filename 
fi

if [ $use_ddp == 1 ]; then
python3 -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=23466 --node_rank=$INDEX \
scripts/evaluation/ddp_wrapper.py \
--module 'style_inference' \
--out_type 'image' \
--adapter_ckpt $adapter_ckpt \
--seed $seed \
--ckpt_path $ckpt \
--base $config \
--savedir $res_dir/$name \
--n_samples $n_samples \
--bs 1 --height 512 --width 512 \
--unconditional_guidance_scale 6.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--filename $filename 
fi