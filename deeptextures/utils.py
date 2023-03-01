"""Utilities.   # 도우미 , 추가기능
"""
import random

import torch
from torchvision.transforms.functional import resize, to_tensor, normalize, to_pil_image
# resize : 입력 이미지를 주어진 크기로 조정 (이미지가 토치 텐서인 경우- […, H, W] 모양)
# to_tensor : PIL img , numpy.ndarray 를 텐서로 변환
# normalize : 평균 및 표준편차로 플로트 텐서 이미지를 정규화. 이 변환은 PIL 이미지를 지원하지 않음.
# to_pil_image : 텐서 또는 ndarray를 PIL 이미지로 변환. 이 기능은 torchscript를 지원하지 않음.

from PIL import Image   # Python Imaging Library (pip install pillow)
# Image 모듈 : 기본적인 이미지 입출력 담당


MEAN = (0.485, 0.456, 0.406)    # 평균
STD = (0.229, 0.224, 0.225) # 표준편차


def set_seed(seed=None):
    """Sets the random seed.    # seed 씨앗 : 랜덤값 줄 때 랜덤 기준값 , 상대가 seed값 알면 랜덤 깨짐
    """
    random.seed(seed)   # 무작위 난수 발생 , seed=None : 현재의 시간이 시드가 됨
    torch.manual_seed(seed) # 난수 생성을 위한 시드를 설정합니다. torch.Generator 객체를 반환합니다
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(device=None):
    """Sets the device.

    by default sets to gpu.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)


def prep_img(image: str, size=None, mean=MEAN, std=STD):
    """Preprocess image.    # 이미지 미리 준비하기
    1) load as PIl
    2) resize
    3) convert to tensor
    4) normalize
    """
    im = Image.open(image)
    #size = size or im.size[::-1]
    width = im.size[0]
    height = im.size[1]
    #texture = resize(im, size)
    texture = resize(im, width*4, height*4)
    texture_tensor = to_tensor(texture).unsqueeze(0)
    texture_tensor = normalize(texture_tensor, mean=mean, std=std)
    return texture_tensor


def denormalize(tensor: torch.Tensor, mean=MEAN, std=STD, inplace: bool = False):
    """Based on torchvision.transforms.functional.normalize.
    """
    tensor = tensor.clone() if not inplace else tensor
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def to_pil(tensor: torch.Tensor):
    """Converts tensor to PIL Image.

    Args:
        tensor (torch.Temsor): input tensor to be converted to PIL Image of torch.Size([C, H, W]).
    Returns:
        PIL Image: converted img.
    """
    img = tensor.clone().detach().cpu()
    img = denormalize(img).clip(0, 1)
    img = to_pil_image(img)
    return img


def to_img(tensor):
    """To image tensor.
    """
    img = tensor.clone().detach().cpu()
    img = denormalize(img).clip(0, 1)
    img = img.permute(1, 2, 0)
    return img
