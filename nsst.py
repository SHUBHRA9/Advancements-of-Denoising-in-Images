import cv2
import numpy as np

# Function to apply weighted bilateral filter
def apply_weighted_bilateral_filter(image):
    # Apply weighted bilateral filter to the image
    filtered_image = cv2.bilateralFilter(image, d=0, sigmaColor=75, sigmaSpace=75)
    return filtered_image

# Function to apply thresholding for high-frequency coefficients
def apply_thresholding(coefficients, threshold):
    # Apply thresholding to high-frequency coefficients
    coefficients[np.abs(coefficients) < threshold] = 0
    return coefficients

# Function to add Gaussian noise to the image
def add_gaussian_noise(image, mean=0, stddev=25):
    noise = np.random.normal(mean, stddev, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

# Function to add salt and pepper noise to the image
def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = np.copy(image)
    salt = np.random.rand(*image.shape) < salt_prob
    pepper = np.random.rand(*image.shape) < pepper_prob
    noisy_image[salt] = 255
    noisy_image[pepper] = 0
    return noisy_image

# Function to add speckle noise to the image
def add_speckle_noise(image, stddev=0.1):
    noise = np.random.normal(0, stddev, image.shape)
    noisy_image = np.clip(image + image * noise, 0, 255).astype(np.uint8)
    return noisy_image

# Function to denoise the image using NSST
def denoise_image(image, noise_std_dev):
    # Apply NSST decomposition to the noisy image
    # Replace the following line with actual NSST decomposition code
    nsst_coefficients = image.copy()

    # Apply weighted bilateral filter to low-frequency coefficients
    low_freq_coefficients = nsst_coefficients.copy()
    denoised_low_freq = apply_weighted_bilateral_filter(low_freq_coefficients)

    # Apply thresholding to high-frequency coefficients
    high_freq_coefficients = nsst_coefficients - denoised_low_freq
    threshold = 2.5 * noise_std_dev  # Adjust this threshold value as needed
    denoised_high_freq = apply_thresholding(high_freq_coefficients, threshold)

    # Combine denoised low and high frequency coefficients
    denoised_coefficients = denoised_low_freq + denoised_high_freq

    # Apply inverse NSST to obtain the final denoised image
    # Replace the following line with actual inverse NSST code
    denoised_image = denoised_coefficients.copy()

    return denoised_image

if __name__ == "__main__":
    # Load the original image
    original_image = cv2.imread('pic1.jpg', cv2.IMREAD_GRAYSCALE)

    if original_image is None:
        print("Error: Failed to load the image")
    else:
        # Add different types of noise to the original image
        noisy_image_gaussian = add_gaussian_noise(original_image)
        noisy_image_salt_pepper = add_salt_and_pepper_noise(original_image)
        noisy_image_speckle = add_speckle_noise(original_image)

        # Set the standard deviation of the noise
        noise_std_dev = 30  # Update this value based on the actual noise level
        # Denoise the noisy images
        denoised_image_gaussian = denoise_image(noisy_image_gaussian, noise_std_dev)
        denoised_image_salt_pepper = denoise_image(noisy_image_salt_pepper, noise_std_dev)
        denoised_image_speckle = denoise_image(noisy_image_speckle, noise_std_dev)

        # Display the original, noisy, and denoised images
        cv2.imshow('Original Image', original_image)
        cv2.imshow('Noisy Image (Gaussian)', noisy_image_gaussian)
        cv2.imshow('Denoised Image (Gaussian)', denoised_image_gaussian)
        cv2.imshow('Noisy Image (Salt and Pepper)', noisy_image_salt_pepper)
        cv2.imshow('Denoised Image (Salt and Pepper)', denoised_image_salt_pepper)
        cv2.imshow('Noisy Image (Speckle)', noisy_image_speckle)
        cv2.imshow('Denoised Image (Speckle)', denoised_image_speckle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
