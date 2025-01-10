from .cl_wrapper import *
from .nodes import *

NODE_CLASS_MAPPINGS = {
    "BlenderValue": BlenderValue,
    "BlenderBrightnessContrast": BlenderBrightnessContrast,
    "BlenderGamma": BlenderGamma,
    "BlenderInvertColor": BlenderInvertColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderValue": "Value",
    "BlenderBrightnessContrast": "Brightness/Contrast",
    "BlenderGamma": "Gamma",
    "BlenderInvertColor": "Invert Color",
}