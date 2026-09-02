import torch

from convert_to_quant.utils.comfy_quant import create_comfy_quant_tensor
from convert_to_quant.utils.tensor_utils import tensor_to_dict


def test_create_comfy_quant_tensor_int4_convrot():
    t = create_comfy_quant_tensor("int4_convrot_w4a4", convrot=True, convrot_groupsize=256)
    cfg = tensor_to_dict(t)
    assert cfg["format"] == "int4_convrot_w4a4"
    assert cfg["convrot"] is True
    assert cfg["convrot_groupsize"] == 256
