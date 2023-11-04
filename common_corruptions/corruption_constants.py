import corruptions

CONSTANTS = {
    'gaussian_noise': corruptions.gaussian_noise,
    'shot_noise': corruptions.shot_noise,
    'impulse_noise': corruptions.impulse_noise,
    'speckle_noise': corruptions.speckle_noise,
    'gaussian_blur': corruptions.gaussian_blur,
    'glass_blur': corruptions.glass_blur,
    'defocus_blur': corruptions.defocus_blur,
    'motion_blur': corruptions.motion_blur,
    'zoom_blur': corruptions.zoom_blur,
    'fog': corruptions.fog,
    'frost': corruptions.frost,
    'snow': corruptions.snow,
    'splatter': corruptions.spatter,
    'contrast': corruptions.contrast,
    'brightness': corruptions.brightness,
    'saturate': corruptions.saturate,
    'jpeg_compression': corruptions.jpeg_compression,
    'pixelate': corruptions.pixelate,
    'elastic': corruptions.elastic_transform
}