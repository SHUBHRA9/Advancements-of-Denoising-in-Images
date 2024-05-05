import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from image_operations import read_image, calculate_psnr
from noise_generation import add_gaussian_noise, add_salt_and_pepper_noise
from denoising import denoise_image_wavelet

def upload_image():
    file_path = filedialog.askopenfilename()
    return file_path

def process_image(img_path, color_mode=True):
    img = read_image(img_path)

    # Add AWGN (Gaussian noise)
    img_awgn = add_gaussian_noise(img)

    # Add salt and pepper noise
    img_salt_pepper = add_salt_and_pepper_noise(img)

    # Denoise images
    img_bayes_awgn = denoise_image_wavelet(img_awgn, method='BayesShrink', wavelet_levels=2)  # Adjust wavelet_levels
    img_visushrink_awgn = denoise_image_wavelet(img_awgn, method='VisuShrink', wavelet_levels=2)  # Adjust wavelet_levels

    img_bayes_salt_pepper = denoise_image_wavelet(img_salt_pepper, method='BayesShrink', wavelet_levels=2)  # Adjust wavelet_levels
    img_visushrink_salt_pepper = denoise_image_wavelet(img_salt_pepper, method='VisuShrink', wavelet_levels=2)  # Adjust wavelet_levels

    # Calculate PSNR
    psnr_awgn_noisy, psnr_awgn_bayes = calculate_psnr(img, img_awgn, img_bayes_awgn)
    psnr_awgn_visu = calculate_psnr(img, img_awgn, img_visushrink_awgn)

    psnr_salt_pepper_noisy, psnr_salt_pepper_bayes = calculate_psnr(img, img_salt_pepper, img_bayes_salt_pepper)
    psnr_salt_pepper_visu = calculate_psnr(img, img_salt_pepper, img_visushrink_salt_pepper)

    # Plotting
    plt.figure(figsize=(20, 20))

    # Original, Noisy AWGN, Noisy Salt & Pepper
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap=plt.cm.gray if not color_mode else None)
    plt.title('Original', fontsize=15)

    plt.subplot(2, 3, 2)
    plt.imshow(img_awgn, cmap=plt.cm.gray if not color_mode else None)
    plt.title('AWGN Noisy', fontsize=15)

    plt.subplot(2, 3, 3)
    plt.imshow(img_salt_pepper, cmap=plt.cm.gray if not color_mode else None)
    plt.title('Salt & Pepper Noisy', fontsize=15)
    
    plt.subplot(2, 3, 4)
    plt.imshow(img_bayes_salt_pepper, cmap=plt.cm.gray if not color_mode else None)
    plt.title('Salt & Pepper Denoised (BayesShrink)', fontsize=15)

    plt.subplot(2, 3, 5)
    plt.imshow(img_visushrink_salt_pepper, cmap=plt.cm.gray if not color_mode else None)
    plt.title('Salt & Pepper Denoised (VisuShrink)', fontsize=15)

    # Denoised AWGN
    

    # New figure for denoised Salt & Pepper
    plt.figure(figsize=(5, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_bayes_awgn, cmap=plt.cm.gray if not color_mode else None)
    plt.title('AWGN Denoised (BayesShrink)', fontsize=10)

    plt.subplot(1, 2, 2)
    plt.imshow(img_visushrink_awgn, cmap=plt.cm.gray if not color_mode else None)
    plt.title('AWGN Denoised (VisuShrink)', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Print PSNR values
    print('AWGN - PSNR Noisy:', psnr_awgn_noisy)
    print('AWGN - PSNR Bayes:', psnr_awgn_bayes)
    print('AWGN - PSNR VisuShrink:', psnr_awgn_visu)
    print('\nSalt & Pepper - PSNR Noisy:', psnr_salt_pepper_noisy)
    print('Salt & Pepper - PSNR Bayes:', psnr_salt_pepper_bayes)
    print('Salt & Pepper - PSNR VisuShrink:', psnr_salt_pepper_visu)

def process_uploaded_image(color_mode=True):
    img_path = upload_image()
    if img_path:
        process_image(img_path, color_mode)

def main():
    root = tk.Tk()
    root.title("Image Processing")
    root.geometry("800x600")  # Resize window to 1000x800

    # Add heading label
    heading_label = tk.Label(root, text="Image Processing", font=("Helvetica", 24))
    heading_label.pack()

    color_button = tk.Button(root, text="Upload Color Image", command=lambda: process_uploaded_image(True))
    color_button.pack()

    grayscale_button = tk.Button(root, text="Upload Grayscale Image", command=lambda: process_uploaded_image(False))
    grayscale_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
