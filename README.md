# Image Denoising with Wavelet Transform

## Overview
In the field of digital image processing, various methodologies are employed to enhance the quality of images. One such technique is image denoising, designed to reduce the noise present in an image while preserving its essential features. This repository introduces a wavelet-based image denoising approach, incorporating techniques like BayesShrink and VisuShrink.

## Methodology
The proposed approach involves the application of wavelet transforms, a mathematical tool that decomposes an image into distinct frequency sub-bands. This enables a more detailed analysis of the image, facilitating effective noise detection and elimination.

Two techniques, BayesShrink and VisuShrink, are utilized in this approach. BayesShrink adapts its threshold level based on the statistical characteristics of the image, making it highly effective in denoising images with varying noise intensities. On the other hand, VisuShrink applies a uniform threshold level across the entire image, simplifying its implementation and accelerating the denoising process.

The denoising procedure operates by applying the wavelet transform to the noisy image, separating it into high-frequency and low-frequency components. The high-frequency components, which primarily contain the noise, undergo thresholding using either BayesShrink or VisuShrink, effectively suppressing the noise. The image is then reconstructed using the inverse wavelet transform, resulting in a denoised image.

## Technologies Used
### Programming Language
- Python

### Libraries
- NumPy: Fundamental package for scientific computing.
- scikit-image (skimage): Python library for image processing.
- matplotlib: Plotting library for visualizations.
- Tkinter: GUI toolkit for creating graphical user interfaces.

## Tools Used
### Integrated Development Environment (IDE)
- Spyder: IDE specialized for scientific computing and data analysis with Python. It offers features such as syntax highlighting, code completion, interactive execution, and debugging.

## Conclusion
This wavelet-based image denoising approach provides a robust and efficient solution for enhancing image quality. Its utilization of BayesShrink and VisuShrink ensures adaptability and speed, making it suitable for a wide range of applications. Future research will focus on optimizing the algorithm and exploring its potential in other domains of image processing.
