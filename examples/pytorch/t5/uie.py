# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This example is used to verify the correctess on summarization task. So, we don't
put benchmark testing in this example.
'''

from __future__ import print_function
import argparse
import json
import numpy as np
import os
import sys

import pandas as pd
import torch
import torch.distributed as dist
from datasets import load_dataset, load_metric
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../../../3rdparty/transformers/src/")

from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from tqdm import tqdm
import configparser
import math
import datetime

from uie.sel2record.record import MapConfig
from uie.sel2record.sel2record import SEL2Record
from uie_utils import post_processing

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_model_location', type=str,
                        default='/models/T5/HF/t5-base/c-models/')
    parser.add_argument('--hf_model_location', type=str,
                        default='/models/T5/HF/t5-base/')
    parser.add_argument('--disable_summarize', action='store_true')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--test_ft', action='store_true')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument("--cache_path", type=str, default="/workdir/datasets/ccdv/")
    parser.add_argument("--max_ite", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--ft_use_hf_config", action="store_true",
                        help="use the hyper-parameters from the hf model")
    parser.add_argument('--lib_path', type=str, default='./lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--rougeLsum_threshold', type=float,
                        help='Threshold of FT rougeLsum score')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=1, help="top k for sampling")
    parser.add_argument("--top_p", type=float, default=0.0, help="top p for sampling")
    parser.add_argument("--beam_width", type=int, default=1, help="beam width for beam search")


    args = parser.parse_args()
    np.random.seed(0) # rouge score use sampling to compute the score

    rank = 0

    disable_summarize = args.disable_summarize
    test_hf = args.test_hf
    test_ft = args.test_ft

    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    ft_model_location = args.ft_model_location + f"/{tensor_para_size}-gpu/"
    hf_model_location = args.hf_model_location

    tokenizer = AutoTokenizer.from_pretrained(hf_model_location)
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = args.batch_size
    # todo
    # dataset_cnn = load_dataset("ccdv/cnn_dailymail", '3.0.0', cache_dir=args.cache_path)

    if rank == 0 and test_hf:
        start_time = datetime.datetime.now()
        if args.data_type == "fp32":
            model = T5ForConditionalGeneration.from_pretrained(hf_model_location, torch_dtype=torch.float32).cuda()
        elif args.data_type == "fp16":
            model = T5ForConditionalGeneration.from_pretrained(hf_model_location, torch_dtype=torch.float16).cuda()
        elif args.data_type == "bf16":
            model = T5ForConditionalGeneration.from_pretrained(hf_model_location, torch_dtype=torch.bfloat16).cuda()
        stop_time = datetime.datetime.now()
        print(f"[INFO] load HF model spend {(stop_time - start_time).total_seconds()} sec")

    if test_ft:
        ckpt_config = configparser.ConfigParser()

        ckpt_config_path = os.path.join(ft_model_location, 'config.ini')
        if os.path.isfile(ckpt_config_path):
            ckpt_config.read(ckpt_config_path)
        else:
            assert False, "[ERROR] This example only support loading model with FT format directly."

        weight_data_type = np.float32
        weight_data_type = {"fp16": np.float16, "fp32": np.float32}[ckpt_config.get("encoder", "weight_data_type")]
        relative_attention_max_distance = 128
        encoder_config = T5Config(vocab_size=ckpt_config.getint("encoder", "vocab_size"),
                                  d_model=ckpt_config.getint("encoder", "d_model"),
                                  d_kv=ckpt_config.getint("encoder", "d_kv"),
                                  d_ff=ckpt_config.getint("encoder", "d_ff"),
                                  num_layers=ckpt_config.getint("encoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint("encoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint("encoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "encoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get("encoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint("encoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint("encoder", "eos_token_id"),
                                  is_gated_act=ckpt_config.getboolean("encoder", "is_gated_act", fallback=0),
                                  )
        decoder_config = T5Config(vocab_size=ckpt_config.getint("decoder", "vocab_size"),
                                  d_model=ckpt_config.getint("decoder", "d_model"),
                                  d_kv=ckpt_config.getint("decoder", "d_kv"),
                                  d_ff=ckpt_config.getint("decoder", "d_ff"),
                                  num_layers=ckpt_config.getint("decoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint("decoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint("decoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "decoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get("decoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint("decoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint("decoder", "eos_token_id"),
                                  decoder_start_token_id=ckpt_config.getint("decoder", "decoder_start_token_id"),
                                  is_gated_act=ckpt_config.getboolean("decoder", "is_gated_act", fallback=0),
                                  )
        assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj
        assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj

        t5_with_bias = ckpt_config.getboolean("structure", "t5_with_bias")
        use_gated_activation = encoder_config.is_gated_act
        position_embedding_type = 0 if ckpt_config.get('structure', 'position_embedding_type') == 'relative' else 1
        activation_type = encoder_config.feed_forward_proj

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1660
        # if tie_word_embeddings == True, scale the decoder output by sequence_output = sequence_output * (self.model_dim**-0.5)
        tie_word_embeddings = ckpt_config.getboolean("decoder", "tie_word_embeddings")
        ft_encoder_weight = FTT5EncoderWeight(
            encoder_config,
            tensor_para_size,
            pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type
        )
        ft_decoding_weight = FTT5DecodingWeight(
            decoder_config,
            tensor_para_size,
            pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type,
        )

        start_time = datetime.datetime.now()
        ft_encoder_weight.load_from_bin(ft_model_location, "Megatron")
        stop_time = datetime.datetime.now()
        print(f"[INFO] load FT encoder model spend {(stop_time - start_time).total_seconds()} sec")
        start_time = datetime.datetime.now()
        ft_decoding_weight.load_from_bin(ft_model_location, "Megatron")
        stop_time = datetime.datetime.now()
        print(f"[INFO] load FT decoding model spend {(stop_time - start_time).total_seconds()} sec")
        if args.data_type == "fp32":
            ft_encoder_weight.to_float()
            ft_decoding_weight.to_float()
        elif args.data_type == "fp16":
            ft_encoder_weight.to_half()
            ft_decoding_weight.to_half()
        elif args.data_type == "bf16":
            ft_encoder_weight.to_bfloat16()
            ft_decoding_weight.to_bfloat16()

        ft_encoder_weight.to_cuda()
        ft_decoding_weight.to_cuda()

        q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
        remove_padding = True
        ft_encoder = FTT5Encoder(ft_encoder_weight.w, args.lib_path, encoder_config.num_heads,
                                 encoder_config.d_kv, encoder_config.d_ff,
                                 encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                 encoder_config.relative_attention_num_buckets,
                                 0, # num_experts
                                 [], # moe_layer_index
                                 relative_attention_max_distance, False, q_scaling, tensor_para_size,
                                 pipeline_para_size, t5_with_bias,
                                 position_embedding_type, moe_k=0, activation_type=activation_type)

        ft_decoding = FTT5Decoding(ft_decoding_weight.w, args.lib_path,
                                   decoder_config.num_heads, decoder_config.d_kv,
                                   decoder_config.d_ff, encoder_config.d_model,
                                   decoder_config.d_model, decoder_config.num_layers,
                                   decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                                   decoder_config.vocab_size, q_scaling,
                                   decoder_config.relative_attention_num_buckets,
                                   0, # num_experts
                                   [], # moe_layer_index
                                   max_distance=relative_attention_max_distance,
                                   tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                                   t5_with_bias=t5_with_bias, position_embedding_type=position_embedding_type,
                                   moe_k=0, activation_type=activation_type, tie_word_embeddings=tie_word_embeddings)

        ft_t5 = FTT5(ft_encoder, ft_decoding)

    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p

    # uie 配置
    map_config = "/workspace/FasterTransformer/examples/pytorch/t5/closest_offset_en.yaml"
    data_folder = "/workspace/FasterTransformer/examples/pytorch/t5/mex_messages"
    map_config = MapConfig.load_from_yaml(map_config)
    schema_dict = SEL2Record.load_schema_dict(data_folder)
    sel2record = SEL2Record(
        schema_dict=schema_dict,
        decoding_schema='spotasoc',
        map_config=map_config,
    )

    def summarize_ft(datapoints):
        lines = [line.strip() for line in datapoints]
        lines = ["<spot> amount<spot> due date<spot> overdue days<asoc> <extra_id_2> " + x for x in lines]
        # line_splits = [line.strip().split() for line in datapoints]


        line_tokens = tokenizer(lines, return_tensors='pt', padding=True)

        with torch.no_grad():
            outputs, ft_output_lens = ft_t5(line_tokens,
                                          None,
                                          beam_width,
                                          args.max_seq_len,
                                          top_k,
                                          top_p,
                                          beam_search_diversity_rate=0.0,
                                          is_return_output_log_probs=False,
                                          len_penalty=1.0,
                                          is_return_cum_log_probs=False)
            # print(output)
            # print(ft_output_len)
        # tokens = [output[0][beam_idx][:ft_output_len[0][beam_idx]] for beam_idx in range(beam_width)]
        """
                result = self._model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self._max_target_length,
        )
        return self._tokenizer.batch_decode(result, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        """

        output_lines = [[tokenizer.decode(output[beam_idx][:ft_output_len[beam_idx]], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        for beam_idx in range(beam_width)][0] for output, ft_output_len in zip(outputs, ft_output_lens)]
        print(output_lines)
        # preds= [post_processing(line) for line in output_lines]
        # records = []
        # for p, text, tokens in zip(preds, lines, line_splits):
        #     r = sel2record.sel2record(pred=p, text=text, tokens=tokens)
        #     records.append(r)
        return output_lines

    def summarize_hf(datapoint):
        line = datapoint.strip()

        line_encoded = tokenizer.encode(line, return_tensors='pt')
        line_encoded = line_encoded.cuda()

        with torch.no_grad():
            if beam_width > 1:
                output = model.generate(line_encoded,
                                        max_length=args.max_seq_len + 1,
                                        num_beams=beam_width,
                                        num_return_sequences=beam_width,
                                        early_stopping=True,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id)
            else:
                output = model.generate(line_encoded,
                                        max_length=args.max_seq_len + 1,
                                        do_sample=True,
                                        top_k=top_k if top_k > 0 else None,
                                        top_p=top_p if top_p > 0.0 else None,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id)
        tokens = [output[beam_idx].cpu().numpy() for beam_idx in range(beam_width)]
        output_lines = [tokenizer.decode(output[beam_idx], skip_special_tokens=True) for beam_idx in range(beam_width)]
        output_lines = [".".join(output_line.split('.')[:4]) + "." for output_line in output_lines]
        return output_lines, tokens

    ft_time = 0.0
    hf_time = 0.0
    df = pd.read_csv("/workspace/FasterTransformer/examples/pytorch/t5/data_pred.csv")
    df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)
    print(df.shape)
    df = df.head()
    data = df["content"].tolist()
    # token_list = [t.strip().split() for t in data]
    # batch_num = math.ceil(len(data) / batch_size)

    def predict(text_list):
        # text_list = [x['text'] for x in read_json_file(gold_filename)]
        # token_list = [x['tokens'] for x in read_json_file(gold_filename)]
        token_list = [t.strip().split() for t in text_list]
        batch_num = math.ceil(len(text_list) / batch_size)
        predict = list()

        for index in tqdm(range(batch_num)):
            start = index * batch_size
            end = index * batch_size + batch_size

            pred_seq2seq = summarize_ft(text_list[start: end])
            pred_seq2seq = [post_processing(x) for x in pred_seq2seq]

            predict += pred_seq2seq

        records = list()
        for p, text, tokens in zip(predict, text_list, token_list):
            r = sel2record.sel2record(pred=p, text=text, tokens=tokens)
            records += [r]

        return records, predict

    # warm up
    _ = predict(data[0:1])
    records, predicts = predict(data)

    offsets = [it["entity"]["offset"] for it in records]
    strings = [it["entity"]["string"] for it in records]
    tokens = ['_'.join((f"<{i}>{v}" for i, v in enumerate(text.split())))
              for text in data]
    df = pd.DataFrame({"content": data, "tokens": tokens, "predict": predicts,
                       "offset": offsets, "strings": strings})
    output_file = "data_pred_result"
    df.to_excel(f"./{output_file}.xlsx", index=None)
    df.to_csv(f"./{output_file}.csv", index=None)

    df.to_excel("./data_pred_result.xlsx", index=None)

    # records = []
    # for i in tqdm(range(len(data))):
    #     datapoint = data[i]
    #     res_ft = summarize_ft(datapoint)
    #     records.append(res_ft)


if __name__ == '__main__':
    main()
