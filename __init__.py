from .cl_wrapper import *
from .nodes import *

# WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "BlenderValue": BlenderValue,
    "BlenderRGB": BlenderRGB,
    #"BlenderBokehImage": BlenderBokehImage,
    "BlenderBrightnessContrast": BlenderBrightnessContrast,
    "BlenderGamma": BlenderGamma,
    #"BlenderHueSaturationValue": BlenderHueSaturationValue,
    "BlenderInvertColor": BlenderInvertColor,
    #"BlenderMix": BlenderMix,
    #"BlenderExposure": BlenderExposure,
    #"BlenderTonemap": BlenderTonemap,
    #"BlenderAlphaOver": BlenderAlphaOver,
    #"BlenderZCombine": BlenderZCombine,
    #"BlenderAlphaConvert": BlenderAlphaConvert,
    #"BlenderConvertColorspace": BlenderConvertColorspace,
    #"BlenderSetAlpha": BlenderSetAlpha,
    "BlenderBlackbody": BlenderBlackbody,
    "BlenderClamp": BlenderClamp,
    #"BlenderCombineColor": BlenderCombineColor,
    "BlenderCombineXYZ": BlenderCombineXYZ,
    "BlenderMapRange": BlenderMapRange,
    #"BlenderMath": BlenderMath,
    "BlenderRGBtoBW": BlenderRGBtoBW,
    "BlenderSeparateColor": BlenderSeparateColor,
    "BlenderSeparateXYZ": BlenderSeparateXYZ,
    #"BlenderVectorMath": BlenderVectorMath,
    #"BlenderWavelength": BlenderWavelength,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderValue": "Value",
    "BlenderRGB": "RGB",
    #"BlenderBokehImage": "Bokeh Image",
    "BlenderBrightnessContrast": "Brightness/Contrast",
    "BlenderGamma": "Gamma",
    #"BlenderHueSaturationValue": "Hue/Saturation/Value"
    "BlenderInvertColor": "Invert Color",
    #"BlenderMix": "Mix"
    #"BlenderExposure": "Exposure",
    #"BlenderTonemap": "Tonemap",
    #"BlenderAlphaOver": "Alpha Over",
    #"BlenderZCombine": "Z Combine",
    #"BlenderAlphaConvert": "Alpha Convert",
    #"BlenderConvertColorspace": "Convert Colorspace",
    #"BlenderSetAlpha": "Set Alpha",
    "BlenderBlackbody": "Blackbody",
    "BlenderClamp": "Clamp",
    #"BlenderCombineColor": "Combine Color",
    "BlenderCombineXYZ": "Combine XYZ",
    "BlenderMapRange": "Map Range",
    #"BlenderMath": "Math",
    "BlenderRGBtoBW": "RGB to BW",
    "BlenderSeparateColor": "Separate Color",
    "BlenderSeparateXYZ": "Separate XYZ",
    #"BlenderVectorMath": "Vector Math",
    #"BlenderWavelength": "Wavelength",
}