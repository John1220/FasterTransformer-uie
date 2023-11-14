# docker
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu22.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list  ##获取ubuntu18.04版本的nvidia-docker列表，结果返回给标准输出


nvidia-docker run -dti --name ft-t5 --restart=always --gpus all --network=host \
--shm-size 5g nvcr.io/nvidia/pytorch:22.09-py3 bash
docker exec -it ft-t5 bash

python3 ../examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
        -saved_dir /workspace/uie-ft-fp16 \
        -in_file /workspace/cpt_latest \
        -inference_tensor_para_size 1 \
        -weight_data_type fp16

python3 ../examples/pytorch/t5/uie.py  \
        --ft_model_location /workspace/uie-ft-fp16 \
        --hf_model_location /workspace/cpt_latest \
        --test_ft


docker run -it --gpus all -p 8900:8900 --name trt-action --shm-size 32G --ulimit memlock=-1 --ulimit stack=67108864 -v /opt/data02/zhan/trt:/work nvcr.io/nvidia/pytorch:23.04-py3 /bin/bash

docker run -it --gpus all -p 8900:8900 --name trt-action --shm-size 32G --ulimit memlock=-1 --ulimit stack=67108864 -v /opt/data02/zhan/trt:/work registry.cn-hangzhou.aliyuncs.com/kristonai/tensorrt_llm:23.11 /bin/bash

'argon2:$argon2id$v=19$m=10240,t=10,p=8$BnMiJsYyKmphGAAeHyMP6g$quecPrkVxhl4OhENoAOKR3ZmGzI/pobpIheAeeISWaI'

nvidia-docker run -dti --name trt_llm --restart=always --gpus all --network=host \
--shm-size 5g registry.cn-hangzhou.aliyuncs.com/kristonai/tensorrt_llm:23.11 bash

https://nanbei.cyou/api/v1/client/subscribe?token=9ab0a833342438ca319dae05b30e2d21

docker exec -it t5_ft bash
nvidia-docker restart d9067f7e718d  --gpus all --network=host

python ../examples/pytorch/t5/translate_example.py \
        --batch_size 32 \
        --beam_width 4 \
        --max_seq_len 128 \
        --data_type fp32 \
        --test_time 0123 \
        --sampling_topk 4 \
        --model t5-small


sudo docker pull nvcr.io/nvidia/pytorch:22.09-py3
docker pull nvcr.io/nvidia/pytorch:23.04-py3

python FasterTransformer/examples/pytorch/t5/translate_example.py \
        --batch_size 32 \
        --beam_width 4 \
        --max_seq_len 128 \
        --data_type fp32 \
        --test_time 0123 \
        --sampling_topk 4 \
        --model /workspace/t5-small-ssm


python3 ../examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
        -saved_dir /workspace/uie-ft-fp16 \
        -in_file /workspace/cpt_latest \
        -inference_tensor_para_size 1 \
        -weight_data_type fp16


python3 ../examples/pytorch/t5/summarization2.py  \
        --ft_model_location /workspace/t5-small-ssm-ft \
        --hf_model_location /workspace/t5-small-ssm \
        --test_ft \
        --test_hf

python3 ../examples/pytorch/t5/uie.py  \
        --ft_model_location /workspace/uie-ft-fp16 \
        --hf_model_location /workspace/cpt_latest \
        --test_ft

        --max_seq_len 512

        --test_hf



<spot> amount<spot> due date<spot> overdue days<asoc> <extra_id_2> [Prestamo rapido]Lleva 2  dias de retraso en su prestamo! El importe total del reembolso es de  487 pesos. Si no paga hoy, su numero de tarjeta se enviara a todas las agencias de credito y se incluira en la lista negra. Esto significa que no puede solicitar un prestamo a ninguna otra empresa financiera durante al menos 12 meses. Por favor, vaya a la aplicacion (Prestamo rapido) o utilice nuestro codigo de pago SPEI (646010262500158686) para realizar su pago ahora.
/workspace/t5-small-ssm-ft/1-gpu//encoder.block.-8.layer.0.layer_norm.weight.bin

root@d9067f7e718d:/workspace/FasterTransformer/build# python3 ../examples/pytorch/t5/summarization2.py  \
>         --ft_model_location /workspace/t5-small-ssm-ft \
>         --hf_model_location /workspace/t5-small-ssm \
>         --test_ft \
>         --test_hf
[INFO] WARNING: Exception occurred in dist.init_process_group(backend = 'mpi'). Maybe the process group has been initialized somewhere else.
Traceback (most recent call last):
  File "../examples/pytorch/t5/summarization2.py", line 407, in <module>
    main()
  File "../examples/pytorch/t5/summarization2.py", line 161, in main
    ft_encoder_weight = FTT5EncoderWeight(
  File "/workspace/FasterTransformer/examples/pytorch/t5/../../../examples/pytorch/t5/utils/ft_encoder.py", line 66, in __init__
    assert world_size == tensor_para_size * \
AssertionError: [ERROR] world_size != tensor_para_size * pipeline_para_size


地址
workspace.featurize.cn
workspace.featurize.cn
端口
37376
37376
用户
featurize
featurize
密码
413f0e51
413f0e51
命令
ssh featurize@workspace.featurize.cn -p 37376


/home/featurize /home/featurize/Services/clash/

[Unit]
Description=clash daemon

[Service]
Type=simple
User=root
ExecStart=/home/featurize/clash -d /home/featurize/Services/clash/
Restart=on-failure

[Install]
WantedBy=multi-user.target


python build.py --model_dir /app/tensorrt_llm/models/cpt_latest/ \
                --use_bert_attention_plugin \
                --use_gpt_attention_plugin \
                --dtype float16 \
                --max_beam_width 3


--quantization int8
ct2-transformers-converter --model cpt_latest/ --quantization int8 --output_dir uie-ct2-int8