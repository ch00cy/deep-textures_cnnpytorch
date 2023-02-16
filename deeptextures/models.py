"""Models.
"""
import torch
import torch.nn as nn
# torch.nn : 인스턴스화 시켜야함 -> attribute(클래스 내부에 포함되어있는 메소드,변수) 활용해 state 저장 가능
# torch.nn.fuctional : 인스턴스화 시킬 필요 없이 바로 입력값 받을 수 있음
from torch.nn.functional import relu    # relu 요소 관련
#from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url  #에러대체
# Pytorch Hub-> 연구 재현성을 촉진하도록 설계된 사전 훈련된 모델 리포지토리
# load_state_dict_from_url : 주어진 URL에서 Torch 직렬화된 개체를 로드 -> Dict[str, Any] 반환
from torchvision.models.vgg import cfgs, make_layers, model_urls


class VGG19(nn.ModuleDict):
    def __init__(self, avg_pool=True):
        super().__init__()
        self.avg_pool = avg_pool

        self.layer_names = """
        conv1_1 conv1_2 pool1
        conv2_1 conv2_2 pool2
        conv3_1 conv3_2 conv3_3 conv3_4 pool3
        conv4_1 conv4_2 conv4_3 conv4_4 pool4
        conv5_1 conv5_2 conv5_3 conv5_4 pool5
        """.split()
        layers = filter(lambda m: not isinstance(m, nn.ReLU), make_layers(cfgs["E"]))
        layers = map(lambda m: nn.AvgPool2d(2, 2) if (isinstance(m, nn.MaxPool2d) and self.avg_pool) else m, layers)
        self.update(dict(zip(self.layer_names, layers)))

        for p in self.parameters():
            p.requires_grad_(False)

    def remap_state_dict(self, state_dict):
        original_names = "0 2 4 5 7 9 10 12 14 16 18 19 21 23 25 27 28 30 32 34 36".split()
        new_mapping = dict(zip(original_names, self.layer_names))
        # Need to copy
        new_state_dict = state_dict.copy()

        for k in state_dict.keys():
            if "classifier" in k:
                del new_state_dict[k]
                continue

            idx = k.split(".")[1]

            name = k.replace("features." + idx, new_mapping[idx])
            new_state_dict[name] = state_dict[k]
            del new_state_dict[k]

        return new_state_dict

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = self.remap_state_dict(state_dict)
        super().load_state_dict(state_dict, **kwargs)

    def forward(self, x, layers: list = None):
        layers = layers or self.keys()
        outputs = {"input": x}
        for name, layer in self.items():
            inp = outputs[[*outputs.keys()][-1]]
            out = relu(layer(inp)) if "pool" not in name else layer(inp)
            outputs.update({name: out})

            del outputs[[*outputs.keys()][-2]]

            if name in layers:
                yield outputs[name]


def vgg19(avg_pool: bool = True, pretrained: bool = True,): # init.py 에서 사용
    model = VGG19(avg_pool=avg_pool)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["vgg19"], progress=True)
        model.load_state_dict(state_dict)
        # model.load_state_dict : 역직렬화된 state_dict를 사용, 모델의 매개변수들을 불러옴.
        # state_dict는 간단히 말해 각 체층을 매개변수 Tensor로 매핑한 Python 사전(dict) 객체.

    return model
