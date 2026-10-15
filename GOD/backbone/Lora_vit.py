
import math
import torch
import copy
from timm.models.layers import DropPath
import timm
from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model
import torch.nn.functional as F
from collections import OrderedDict
import torch

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
class LinearList(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        
        self.fan_in_fan_out = fan_in_fan_out
        self.lora_As = nn.ParameterList()  # 所有任务的A矩阵
        self.lora_Bs = nn.ParameterList()  # 所有任务的B矩阵
        self.lora_emaA = nn.Parameter(
            self.weight.new_zeros((r, self.in_features)),
            requires_grad=False  # 关键：直接禁用梯度
        )
        self.lora_emaB = nn.Parameter(
            self.weight.new_zeros((self.out_features, r)),
            requires_grad=False  # 关键：直接禁用梯度
        )
        self.task_scaling_EMA = lora_alpha / r
        self.task_scalings = []  # 每个任务的缩放因子
        # 冻结预训练权重
        self.weight.requires_grad = False
        self.reset_parameters()
        self.cur_loraid = self.num_lora_modules()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
    def startEMA(self):
        # 深拷贝数据（独立内存）
        self.lora_emaA.data.copy_(self.lora_As[-1].data)
        self.lora_emaB.data.copy_(self.lora_Bs[-1].data)
    def EMA(self, alpha):
        with torch.no_grad():  # 禁用梯度计算（EMA不需要梯度）
            self.lora_emaA.data.copy_(alpha * self.lora_emaA.data + (1.0 - alpha) * self.lora_As[-1].data)
            self.lora_emaB.data.copy_(alpha * self.lora_emaB.data + (1.0 - alpha) * self.lora_Bs[-1].data)
    def num_lora_modules(self) -> int:
        """获取当前已注册的LoRA任务数量"""
        return len(self.lora_As)

    def freeze_except_current(self):
        """冻结除最后一个LoRA模块外的所有LoRA模块（核心功能）"""
        last_idx = len(self.lora_As) - 1  # 获取最后一个LoRA模块的索引
        
        for i in range(len(self.lora_As)):
            if i != last_idx:  # 非最后一个模块
                self.lora_As[i].requires_grad_(False)
                self.lora_Bs[i].requires_grad_(False)
            else:  # 最后一个模块
                self.lora_As[i].requires_grad_(True)
                self.lora_Bs[i].requires_grad_(True)

    def add_task(self, r: int = None, lora_alpha: int = None):

        r = r or self.r
        lora_alpha = lora_alpha or self.lora_alpha
        scaling = lora_alpha / r

        lora_A = nn.Parameter(self.weight.new_zeros((r, self.in_features)))
        lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, r)))

        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        nn.init.zeros_(lora_B)
        if(len(self.lora_As) > 0):
            lora_A = copy.deepcopy(self.lora_As[-1])
            lora_B = copy.deepcopy(self.lora_Bs[-1])
        self.lora_As.append(lora_A)
        self.lora_Bs.append(lora_B)
        self.task_scalings.append(scaling)
        self.cur_loraid = len(self.lora_As) - 1
        self.freeze_except_current()
        return len(self.lora_As) - 1

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w  
        nn.Linear.train(self, mode)
    
    def forward(self, x: torch.Tensor, lora_ids: list = [0],ifEMA= False):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        lora_output = torch.zeros_like(result)
        if(ifEMA):
            lora_A = self.lora_emaA
            lora_B = self.lora_emaB
            scaling = self.task_scaling_EMA
            lora_output += self.lora_dropout(x) @ lora_A.transpose(0, 1) @ lora_B.transpose(0, 1)
            lora_output = lora_output * scaling
            result += lora_output
            return {'result': result, 'loraValue': lora_output}
        else:
            for lora_id in lora_ids:
                if 0 <= lora_id < len(self.lora_As):
                    lora_A = self.lora_As[lora_id]
                    lora_B = self.lora_Bs[lora_id]
                    scaling = self.task_scalings[lora_id]

                    # 应用LoRA适配器
                    lora_output += self.lora_dropout(x) @ lora_A.transpose(0, 1) @ lora_B.transpose(0, 1)
                    lora_output = lora_output * scaling
                    result += lora_output

            return    {'result': result,  'loraValue': lora_output  }
    def freeze_lora(self, lora_ids: list, A_loras: bool,B_loras:bool):
        """
        冻结指定的 LoRA 模块的参数（lora_A 和 lora_B）。
        :param freeze_lora_ids: 需要冻结的 LoRA 模块的索引列表
        """
        for lora_id in lora_ids:
            if 0 <= lora_id < len(self.lora_As):
                if(A_loras):self.lora_As[lora_id].requires_grad = False
                if(B_loras):self.lora_Bs[lora_id].requires_grad = False
            else:
                raise ValueError(f"Invalid lora_id {lora_id}: it must be between 0 and {len(self.lora_As)-1}.")

    def unfreeze_lora(self, lora_ids: list):
        for lora_id in lora_ids:
            if 0 <= lora_id < len(self.lora_As):
                self.lora_As[lora_id].requires_grad = True
                self.lora_Bs[lora_id].requires_grad = True
            else:
                raise ValueError(f"Invalid lora_id {lora_id}: it must be between 0 and {len(self.lora_As) - 1}.")

    def copy_lora_weights(self, loraidA: int, loraidB: int,A_loras: bool,B_loras:bool):
        if 0 <= loraidA < len(self.lora_As) and 0 <= loraidB < len(self.lora_As):
            # 将 loraidA 的权重赋值给 loraidB
            if(A_loras):self.lora_As[loraidB] = self.lora_As[loraidA].clone()
            if(B_loras):self.lora_Bs[loraidB] = self.lora_Bs[loraidA].clone()
        else:
            raise ValueError(f"Invalid lora_id. loraidA should be between 0 and {len(self.lora_As)-1}, "
                            f"and loraidB should be between 0 and {len(self.lora_As)-1}.")

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., config=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.config =config
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_proj = LinearList(dim, dim, bias=qkv_bias, r=config.r, lora_alpha=config.lora_alpha, 
                             lora_dropout=config.lora_dropout, fan_in_fan_out=False)
        self.k_proj = LinearList(dim, dim, bias=qkv_bias, r=config.r, lora_alpha=config.lora_alpha, 
                             lora_dropout=config.lora_dropout, fan_in_fan_out=False)
        self.v_proj = LinearList(dim, dim, bias=qkv_bias, r=config.r, lora_alpha=config.lora_alpha, 
                             lora_dropout=config.lora_dropout, fan_in_fan_out=False)
        self.cur_loraid = 0
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    def add_task(self):
        self.cur_loraid = self.q_proj.add_task()
        self.k_proj.add_task()
        self.v_proj.add_task()
    def startEMA(self):
        # 深拷贝数据（独立内存）
        self.q_proj.startEMA()
        self.k_proj.startEMA()
        self.v_proj.startEMA()
    def EMA(self,alpha):
        # 深拷贝数据（独立内存）
        self.q_proj.EMA(alpha)
        self.k_proj.EMA(alpha)
        self.v_proj.EMA(alpha)

    def freeze_lora(self,lora_ids: list, A_loras: bool,B_loras:bool):
        self.q_proj.freeze_lora(lora_ids,A_loras,B_loras)
        self.k_proj.freeze_lora(lora_ids,A_loras,B_loras)
        self.v_proj.freeze_lora(lora_ids,A_loras,B_loras)
    def unfreeze_lora(self,lora_ids: list):
        self.q_proj.unfreeze_lora(lora_ids)
        self.k_proj.unfreeze_lora(lora_ids)
        self.v_proj.unfreeze_lora(lora_ids)
    def copy_lora_weights(self, loraidA: int, loraidB: int,A_loras: bool,B_loras:bool):
        self.q_proj.copy_lora_weights(loraidA,loraidB,A_loras,B_loras)
        self.k_proj.copy_lora_weights(loraidA,loraidB,A_loras,B_loras)
        self.v_proj.copy_lora_weights(loraidA,loraidB,A_loras,B_loras)   

    def forward(self, x,lora_ids=[0],ifEMA=False):
        B, N, C = x.shape

        q = self.q_proj(x,lora_ids,ifEMA)['result']
        k = self._shape(self.k_proj(x,lora_ids,ifEMA)['result'], -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x,lora_ids,ifEMA)['result'], -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,config=self.config)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.cur_loraid = 0
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)
    def unfreeze_lora(self,lora_ids):
        self.attn.unfreeze_lora(lora_ids)
    def add_task(self):
        self.cur_loraid = self.attn.add_task()
    def freeze_lora(self,lora_ids: list, A_loras: bool,B_loras:bool):
        self.attn.freeze_lora(lora_ids,A_loras,B_loras)
    def startEMA(self):
        # 深拷贝数据（独立内存）
        self.attn.startEMA()
    def EMA(self,alpha):
        # 深拷贝数据（独立内存）
        self.attn.EMA(alpha)
    def copy_lora_weights(self, loraidA: int, loraidB: int,A_loras: bool,B_loras:bool):
        self.attn.copy_lora_weights(loraidA,loraidB,A_loras,B_loras)

    def forward(self, x,lora_ids=[0],ifEMA=False):
        x = x + self.drop_path(self.attn(self.norm1(x),lora_ids,ifEMA))
        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))
        x = residual + x
        return x



class VisionTransformer(nn.Module):
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()


        print("I'm using ViT.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.cur_loraid = 0
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.select=9
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
    def init_weights(self, mode=''):
        raise NotImplementedError()
    def add_task(self):
        for block in self.blocks:
            self.cur_loraid =block.add_task()
    def freeze_lora(self,lora_ids: list, A_loras: bool,B_loras:bool):
        for block in self.blocks:
            block.freeze_lora(lora_ids,A_loras,B_loras)

    def unfreeze_lora(self,lora_ids,select=9):
        lengh1 = select
        self.select = lengh1
        for block in self.blocks[lengh1:]:
            block.unfreeze_lora(lora_ids)
        print(lengh1)
    def startEMA(self):
        for block in self.blocks:
            block.startEMA()
    def EMA(self,alpha):
        for block in self.blocks:
            block.EMA(alpha)
    def copy_lora_weights(self, loraidA: int, loraidB: int,A_loras: bool,B_loras:bool):
        for block in self.blocks:
            block.copy_lora_weights(loraidA,loraidB,A_loras,B_loras)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x,lora_ids = [0],ifEMA=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        SL_x = None
        for idx, blk in enumerate(self.blocks):
            x = blk(x,lora_ids,ifEMA)
            if (ifEMA and idx == self.select-1):
                SL_x = x
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome,SL_x

    def forward(self, x,lora_ids=[0],ifEMA=False):
        x,SL_x = self.forward_features(x,lora_ids,ifEMA)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x,SL_x
    def forward_SL(self,x,lora_ids):
        for idx, blk in enumerate(self.blocks[self.select:]):
            x = blk(x,lora_ids)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x






def vit_base_patch16_224(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    checkpoint_model=timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    print(model)
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 


    return model



def vit_base_patch16_224_in21k_adapter(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    checkpoint_model=timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model

