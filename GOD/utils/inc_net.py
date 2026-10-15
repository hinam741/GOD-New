import copy
import logging
import torch
from torch import nn
from backbone.linears import ArcFace,Multi_lora_etf
import timm

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    if '_lora' in name:
        if args["model_name"] ==  "GOD":
            from backbone import Lora_vit
            from easydict import EasyDict
            tuning_config = EasyDict(
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                r=16, lora_alpha=1, lora_dropout=0.
            )
            if name == "vit_base_patch16_224_lora":
                model = Lora_vit.vit_base_patch16_224(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
class IncrementalNet_ETF_mutli_lora(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()
        self.cosine_fc = None
        self.simple_fc = None
        self.args = args
        self._cur_task = -1

    def update_fc(self, nb_classes, Hiddensize, increment):
        fc = self.generate_ETFfc(self.feature_dim, nb_classes, Hiddensize, increment)
        self.fc = fc
    def generate_ETFfc(self, in_dim, out_dim, Hiddensize, increment):
        fc = Multi_lora_etf(out_dim, in_dim, Hiddensize, increment)
        return fc
    def generate_simplefc(self, in_dim, out_dim):
        fc = ArcFace(in_dim, out_dim,m=self.args["m"])
        return fc
    def update_simplefc(self, nb_classes):
        fc = self.generate_simplefc(self.feature_dim, nb_classes)
        del self.simple_fc
        self.simple_fc = fc

    def forward(self, x, loraid,Train=False):
        if self.model_type == 'cnn':
            for  i in range(loraid):
                x = self.backbone(x, [i])
                out = self.fc(x["features"])
                out.update(x)
        else:
            output={}
            logits = []
            if(Train==False):
                for i in range(loraid+1):
                    feature,_ = self.backbone(x, [i])
                    out = self.fc(feature,i)
                    logits.append(out)
                logits = torch.cat(logits, dim=1)
                output["logits"] = logits
            else:
                feature,_ = self.backbone(x, [loraid])
                out = self.fc(feature, loraid)
                output["logits"] = out
                output["feature"] = feature
        return output

    def forward_EMA(self, x):
        output={}
        logits = []
        feature,SL_x = self.backbone(x, [],True)
        for i in range(self._cur_task+1):
            out = self.fc(feature,i)
            logits.append(out)
        logits = torch.cat(logits, dim=1)
        output["logits"] = logits
        output["SL_x"] = SL_x
        return output
    def forwardnew(self, x, loraids):
        output = {}
        logits = []
        for i in range(self._cur_task + 1):
            if i in loraids:
                feature = self.backbone.forward_SL(x, [i])
                out = self.fc(feature, i)
                logits.append(out)
            else:
                out = torch.full((1, self.args["init_cls"]), float('-inf')).to(self._device)
                logits.append(out)
        logits = torch.cat(logits, dim=1)
        output["logits"] = logits
        return output

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x, loraids)['features']
        else:
            return self.backbone(x, loraids)


