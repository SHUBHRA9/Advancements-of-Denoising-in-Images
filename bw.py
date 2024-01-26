import matplotlib.pyplot as plt
from skimage.restoration import(denoise_wavelet,estimate_sigma)
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import skimage.io

img=skimage.io.imread("pic1.jpg")
img=skimage.img_as_float(img)
sigma=0.1
imgn=random_noise(img, var=sigma**2)

sigma_est=estimate_sigma(imgn,average_sigmas=True)

img_bayes=denoise_wavelet(imgn,method="BayesShrink", mode='soft', wavelet_levels=3,wavelet='bior6.8', rescale_sigma=True)


img_visushrink=denoise_wavelet(imgn,method='VisuShrink',mode='soft',sigma=sigma_est/3,wavelet_levels=5,wavelet='bior6.8',rescale_sigma=True)

psnr_noisy=peak_signal_noise_ratio(img, imgn)
psnr_bayes=peak_signal_noise_ratio(img, img_bayes)
psnr_visu=peak_signal_noise_ratio(img, img_visushrink)

plt.figure(figsize=(30,30))

plt.subplot(2,2,1)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original Image', fontsize=30)

plt.subplot(2,2,2)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Noisy Image', fontsize=30)

plt.subplot(2,2,3)
plt.imshow(img_bayes, cmap=plt.cm.gray)
plt.title('Denoised Image using Bayes', fontsize=30)

plt.subplot(2,2,4)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Denoised Image using Visushrink', fontsize=30)

plt.show()

print('PSNR(Original vs Noisy Image): ',psnr_noisy)
print('PSNR(Original vs Denoised (Visushrink)): ',psnr_visu)
print('PSNR(Original vs Denoised (Bayes)): ',psnr_bayes)