import numpy as np
import skimage.color
from skimage.restoration import denoise_wavelet, estimate_sigma

def denoise_image_wavelet(image, method='BayesShrink', wavelet_levels=3, sigma=None, color_mode=True):
    
    if not color_mode:
        if method == 'BayesShrink':
            return denoise_wavelet(image, method='BayesShrink', mode='soft', wavelet_levels=wavelet_levels)
        elif method == 'VisuShrink':
            sigma_est = estimate_sigma(image, average_sigmas=True)
            if sigma is None:
                sigma = sigma_est / 3
            return denoise_wavelet(image, method='VisuShrink', mode='soft', sigma=sigma, wavelet_levels=wavelet_levels)
    else:
        # Convert image to YUV color space
        yuv_image = skimage.color.rgb2yuv(image)

        # Denoise luminance component (Y channel)
        y_channel = yuv_image[..., 0]
        y_denoised = denoise_wavelet(y_channel, method=method, mode='soft', wavelet_levels=wavelet_levels)

        # Combine denoised luminance channel with original chrominance channels (UV channels)
        denoised_yuv_image = np.stack((y_denoised, yuv_image[..., 1], yuv_image[..., 2]), axis=-1)

        # Convert back to RGB color space
        return skimage.color.yuv2rgb(denoised_yuv_image)
