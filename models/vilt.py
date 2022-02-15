import math
from turtle import xcor
from typing import Callable

import inspect
import torch
from colossalai import nn as col_nn
from colossalai.registry import LAYERS, MODELS
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.builder.pipeline import partition_uniform
from torch import dtype, nn
from model_zoo.vit.vit import ViTBlock, ViTEmbedding
from utils import heads, objectives
import torch.nn.functional as F
import torch
import torch.nn as nn
import inspect
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.nn.layer.colossalai_layer import LayerNorm
# from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
from colossalai.logging import get_dist_logger
from colossalai.builder.pipeline import partition_uniform
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from colossalai.builder import build_pipeline_model

@MODELS.register_module

class ViLT(nn.Module):
    def __init__(self,
                config,
                # vocab_size,
                # hidden_size,
                # max_sequence_length,
                # num_attettion_heads,
                # num_layers,
                # add_binary_head,
                # is_naive_fp16,
                img_size: int = 384,
                patch_size: int = 16,
                in_chans: int = 3,
                num_classes: int = 1000,
                depth: int = 12,
                num_heads: int = 12,
                dim: int = 768,
                mlp_ratio: int = 4,
                attention_dropout: float = 0.,
                dropout: float = 0.1,
                dropout_prob=0.1,
                drop_path: float = 0.,
                init_std=0.02,
                layernorm_epsilon: float = 1e-6,
                activation: Callable = nn.functional.gelu,
                representation_size: int = None,
                convert_fp16_to_fp32_in_softmax=False,
                dtype: dtype = None,
                bias: bool = True,
                checkpoint: bool = False,
                init_method: str = 'torch',
                first_stage=True,
                last_stage=True,
                start_idx=0,
                end_idx=None,):
        super().__init__()
        max_sequence_length = config["max_text_len"]
        num_layers = config["num_layers"]
        vocab_size = config["vocab_size"]
        self.vocab_size = vocab_size
        hidden_size = config["hidden_size"]
        self.first_stage = first_stage
        self.last_stage = last_stage
        # self.seq_parallel_size = gpc.get_world_size(ParallelMode.SEQUENCE)
        # assert max_sequence_length % self.seq_parallel_size == 0, 'sequence length is not divisible by the sequence parallel size'
        # self.sub_seq_length = max_sequence_length // self.seq_parallel_size
        self.init_std = init_std
        self.num_layers = num_layers

        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * mlp_ratio,
            max_position_embeddings=max_sequence_length,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

        self.pooler = heads.Pooler(hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)
        # self.text_embedding = Embedding(hidden_size=hidden_size,
        #                             vocab_size=vocab_size,
        #                             max_sequence_length=max_sequence_length,
        #                             embedding_dropout_prob=dropout_prob,
        #                             num_tokentypes=num_tokentypes)
        self.text_embedding = BertEmbeddings(bert_config)
        self.vis_embedding = ViTEmbedding(img_size=img_size,
                                patch_size=patch_size,
                                in_chans=in_chans,
                                embedding_dim=dim,
                                dropout=dropout,
                                dtype=dtype,
                                init_method=init_method)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        blocks = [
            ViTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                drop_path=dpr[i],
                activation=activation,
                dtype=dtype,
                bias=bias,
                checkpoint=checkpoint,
                init_method=init_method,
            ) for i in range(depth)
        ]
        norm = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        # self.mlm_score = BertDualHead(hidden_size, self.text_embedding.word_embedding_weight.size(0),
        #                         add_binary_head=add_binary_head)
        # self.head = self.mlm_score


        # # transformer layers
        # self.bert_layers = nn.ModuleList()

        # if start_idx is None and end_idx is None:
        #     start_idx = 0
        #     end_idx = num_layers

        # for i in range(start_idx, end_idx):
        #     bert_layer = BertLayer(layer_number=i+1,
        #                            hidden_size=hidden_size,
        #                            num_attention_heads=num_attettion_heads,
        #                            attention_dropout=dropout_prob,
        #                            mlp_ratio=mlp_ratio,
        #                            hidden_dropout=dropout_prob,
        #                            convert_fp16_to_fp32_in_softmax=convert_fp16_to_fp32_in_softmax,
        #                            is_naive_fp16=is_naive_fp16
        #                            )
        #     self.bert_layers.append(bert_layer)

        if self.last_stage:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)
        # self.reset_parameters()

        self.layer_norm = LayerNorm(hidden_size)

        layers=[]
        layers.extend(blocks)
        layers.extend([norm])
        self.layers = nn.Sequential(
            *layers
        )
        # self.layers = build_pipeline_model(self.layers, num_chunks=1, verbose=True)


    def infer(self,x,image_token_type_idx=1):
        do_mlm = "_mlm"
        if f"image_{image_token_type_idx - 1}" in x:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"
        img = x[imgkey]
        text_ids = x[f"text_ids{do_mlm}"]
        text_labels = x[f"text_labels{do_mlm}"]
        image_embeds = self.vis_embedding(img)
        text_embeds = self.text_embedding(text_ids)
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        x = co_embeds
        x = self.layers(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            # "image_labels": image_labels,
            # "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            # "text_masks": text_masks,
            # "patch_index": patch_index,
        } 
        return ret

    def forward(self,x):
        ret = dict()
        ret.update(self.compute_mlm(x))
        return ret

    def compute_mlm(self, batch):
        infer = self.infer(batch)
        mlm_logits = self.mlm_score(infer["text_feats"])
        mlm_labels = infer["text_labels"]

        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, self.vocab_size),
            mlm_labels.view(-1),
            ignore_index=-100,
        )

        ret = {
            "mlm_loss": mlm_loss,
            "mlm_logits": mlm_logits,
            "mlm_labels": mlm_labels,
            "mlm_ids": infer["text_ids"],
        }

        # phase = "train" if pl_module.training else "val"
        # loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
        # acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        #     ret["mlm_logits"], ret["mlm_labels"]
        # )

        return ret

    # def _init_normal(self, tensor):
    #     init_normal(tensor, sigma=self.init_std)

    # def _output_init_normal(self, tensor):
    #     output_init_normal(tensor, sigma=self.init_std, num_layers=self.num_layers)

    # def reset_parameters(self):
    #     # initialize embedding
    #     self._init_normal(self.text_embedding.word_embedding_weight)
    #     self._init_normal(self.text_embedding.position_embeddings.weight)
    #     if self.text_embedding.tokentype_embeddings:
    #         self._init_normal(self.text_embedding.tokentype_embeddings.weight)

    #     # intiailize bert layer
    #     for layer in self.bert_layers:
    #         # initialize self attention
    #         self._init_normal(layer.self_attention.query_key_value.weight)
    #         self._output_init_normal(layer.self_attention.dense.weight)
    #         self._init_normal(layer.mlp.dense_h_to_4h.weight)
    #         self._output_init_normal(layer.mlp.dense_4h_to_h.weight)

def _filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


# def build_pipeline_bert(num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
#     logger = get_dist_logger()
#     pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
#     pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
#     rank = gpc.get_global_rank()
#     wrapper = PipelineSharedModuleWrapper([0, pipeline_size - 1])
#     parts = partition_uniform(num_layers, pipeline_size, num_chunks)[pipeline_rank]
#     models = []
#     for start, end in parts:
#         kwargs['num_layers'] = num_layers
#         kwargs['start_idx'] = start
#         kwargs['end_idx'] = end
#         kwargs['first_stage'] = start == 0
#         kwargs['last_stage'] = end == num_layers
#         logger.info(f'Rank{rank} build layer {start}-{end}, {end-start}/{num_layers} layers')
#         chunk = PipelineBertForPretrain(**_filter_kwargs(PipelineBertForPretrain.__init__, kwargs)).to(device)
#         if start == 0:
#             wrapper.register_module(chunk.embedding.word_embeddings)
#         elif end == num_layers:
#             wrapper.register_module(chunk.word_embeddings)
#         models.append(chunk)
#     if len(models) == 1:
#         model = models[0]
#     else:
#         model = nn.ModuleList(models)
#     return model


def _filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _build_pipeline_vit(module_cls, num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
    logger = get_dist_logger()
    if gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    else:
        pipeline_size = 1
        pipeline_rank = 0
    rank = gpc.get_global_rank()
    parts = partition_uniform(num_layers, pipeline_size, num_chunks)[pipeline_rank]
    models = []

    for start, end in parts:
        kwargs['first_stage'] = start == 0
        kwargs['last_stage'] = end == num_layers
        kwargs['start_idx'] = start
        kwargs['end_idx'] = end
        logger.info(f'Rank{rank} build layer {start}-{end}, {end-start}/{num_layers} layers')
        chunk = module_cls(**_filter_kwargs(module_cls.__init__, kwargs)).to(device)
        models.append(chunk)
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)
    return model


# def build_pipeline_vit(num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
#     return _build_pipeline_vit(PipelineVisionTransformer, num_layers, num_chunks, device, **kwargs)




def get_current_device():
    '''
    Returns the index of a currently selected device (gpu/cpu).
    '''
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return 'cpu'