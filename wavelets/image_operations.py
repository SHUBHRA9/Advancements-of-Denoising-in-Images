import skimage.io
from skimage.metrics import peak_signal_noise_ratio

def read_image(image_path):
    return skimage.io.imread(image_path)

def calculate_psnr(original_img, noisy_img, denoised_img):
    psnr_noisy = peak_signal_noise_ratio(original_img, noisy_img)
    psnr_denoised = peak_signal_noise_ratio(original_img, denoised_img)
    return psnr_noisy, psnr_denoised
