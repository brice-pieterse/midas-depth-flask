import cv2
import torch
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose

default_models = {
    "dpt_beit_large_512": "midas/models/dpt_beit_large_512.pt",
    "dpt_beit_large_384": "midas/models/dpt_beit_large_384.pt",
    "dpt_beit_base_384": "midas/models/dpt_beit_base_384.pt",
    "dpt_swin2_large_384": "midas/models/dpt_swin2_large_384.pt",
    "dpt_swin2_base_384": "midas/models/dpt_swin2_base_384.pt",
    "dpt_swin2_tiny_256": "midas/models/dpt_swin2_tiny_256.pt",
    "dpt_swin_large_384": "midas/models/dpt_swin_large_384.pt",
    "dpt_next_vit_large_384": "midas/models/dpt_next_vit_large_384.pt",
    "dpt_levit_224": "midas/models/dpt_levit_224.pt",
    "dpt_large_384": "midas/models/dpt_large_384.pt",
    "dpt_hybrid_384": "midas/models/dpt_hybrid_384.pt",
    "midas_v21_384": "midas/models/midas_v21_384.pt",
    "midas_v21_small_256": "midas/models/midas_v21_small_256.pt",
    "openvino_midas_v21_small_256": "midas/models/openvino_midas_v21_small_256.xml",
}


def load_model(device, model_path="midas/models/dpt_swin2_large_384.pt", model_type="dpt_swin2_large_384", optimize=True, height=None, square=False):
    """Load the specified network.

    Args:
        device (device): the torch device used
        model_path (str): path to saved model
        model_type (str): the type of the model to be loaded
        optimize (bool): optimize the model to half-integer on CUDA?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?

    Returns:
        The loaded network, the transform which prepares images as input to the network and the dimensions of the
        network input
    """

    keep_aspect_ratio = not square

    if model_type == "dpt_beit_large_512":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitl16_512",
            non_negative=True,
        )
        net_w, net_h = 512, 512
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_beit_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_beit_base_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitb16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2l24_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_base_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2b24_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_tiny_256":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2t16_256",
            non_negative=True,
        )
        net_w, net_h = 256, 256
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


    if height is not None:
        net_w, net_h = height, height

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()
    model.to(device)

    return model, transform, net_w, net_h