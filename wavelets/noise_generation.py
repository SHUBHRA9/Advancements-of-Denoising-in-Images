import numpy as np

def add_gaussian_noise(image, sigma=0.1):

    noise = np.random.normal(scale=sigma, size=image.shape)
    noisy_image = np.clip(image + noise, 0, 255)

    return noisy_image.astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_vs_pepper=0.5, amount=0.01, color_mode=True):
    
    noisy_image = np.copy(image)

    if not color_mode:
        # Grayscale image
        # Salt noise
        num_salt = np.ceil(amount * image.size * salt_vs_pepper)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 255

        # Pepper noise
        num_pepper = np.ceil(amount * image.size * (1. - salt_vs_pepper))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = 0
    else:
        # Color image
        # Salt noise
        num_salt = np.ceil(amount * image[..., 0].size * salt_vs_pepper)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image[..., 0].shape]
        for i in range(image.shape[-1]):
            noisy_image[..., i][tuple(coords)] = 255

        # Pepper noise
        num_pepper = np.ceil(amount * image[..., 0].size * (1. - salt_vs_pepper))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image[..., 0].shape]
        for i in range(image.shape[-1]):
            noisy_image[..., i][tuple(coords)] = 0

    return noisy_image.astype(np.uint8)
