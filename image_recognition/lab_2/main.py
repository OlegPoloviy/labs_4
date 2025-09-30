import cv2
import numpy as np
from matplotlib import pyplot as plt

# --- PREPARATION OF THE COLOR IMAGE ---

# 1. Load the image in color (default)
# Note: OpenCV loads images in BGR channel order
image_bgr = cv2.imread('D:/projects/labs/image_recognition/lab_2/obamna.png') 

if image_bgr is None:
    print("Error with receiving a file.")
else:
    # 2. Convert BGR to RGB for correct display with matplotlib
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    height, width, channels = image_rgb.shape

    # 4. Show the original image for comparison
    plt.imshow(image_rgb)
    plt.title('Оригінальне кольорове зображення')
    plt.show()

sharpen_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

# Create an empty array for the result with three channels
sharpened_image_manual = np.zeros_like(image_rgb)

# Iterate over each channel (0=R, 1=G, 2=B)
for c in range(channels):
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Extract a 3x3 region from the current channel
            region = image_rgb[y-1:y+2, x-1:x+2, c]
            
            # Convolution operation
            new_pixel_value = np.sum(region * sharpen_kernel)
            
            # Clip the value to valid range
            sharpened_image_manual[y, x, c] = np.clip(new_pixel_value, 0, 255)

# Display the result
plt.imshow(sharpened_image_manual)
plt.title('Фільтр чіткості (кольоровий)')
plt.show()


blur_kernel = np.array([
    [0.000789, 0.006581, 0.013347, 0.006581, 0.000789],
    [0.006581, 0.054901, 0.111345, 0.054901, 0.006581],
    [0.013347, 0.111345, 0.225821, 0.111345, 0.013347],
    [0.006581, 0.054901, 0.111345, 0.054901, 0.006581],
    [0.000789, 0.006581, 0.013347, 0.006581, 0.000789]
])

blurred_image_manual = np.zeros_like(image_rgb)
offset_blur = 2 # radius for 5x5 kernel

for c in range(channels):
    for y in range(offset_blur, height - offset_blur):
        for x in range(offset_blur, width - offset_blur):
            # Extract a 5x5 region from the current channel
            region = image_rgb[y-offset_blur:y+offset_blur+1, x-offset_blur:x+offset_blur+1, c]
            
            # Convolution operation
            new_pixel_value = np.sum(region * blur_kernel)
            
            blurred_image_manual[y, x, c] = np.clip(new_pixel_value, 0, 255)

# Display the result
plt.imshow(blurred_image_manual)
plt.title('Фільтр розмиття (кольоровий)')
plt.show()

median_filtered_manual = np.zeros_like(image_rgb)
window_size = 3
offset_median = window_size // 2

for c in range(channels):
    for y in range(offset_median, height - offset_median):
        for x in range(offset_median, width - offset_median):
            # Extract a window from the current channel
            region = image_rgb[y-offset_median:y+offset_median+1, x-offset_median:x+offset_median+1, c]
            
            # Compute the median
            median_value = np.median(region)
            
            median_filtered_manual[y, x, c] = median_value

# Display the result
plt.imshow(median_filtered_manual)
plt.title('Медіанний фільтр (кольоровий)')
plt.show()


def create_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi=0):
    """
    Create a Gabor filter kernel manually.
    ksize: kernel size (e.g., 31)
    sigma: standard deviation of the Gaussian envelope
    theta: orientation (angle in radians)
    lambd: wavelength of the sinusoidal factor
    gamma: spatial aspect ratio
    psi: phase offset
    """
    # Створюємо сітку координат для ядра
    half_ksize = ksize // 2
    x_coords, y_coords = np.meshgrid(np.arange(-half_ksize, half_ksize + 1),
                                     np.arange(-half_ksize, half_ksize + 1))
    
    # Rotate coordinates according to angle theta (computes x' and y')
    x_prime = x_coords * np.cos(theta) + y_coords * np.sin(theta)
    y_prime = -x_coords * np.sin(theta) + y_coords * np.cos(theta)
    
    # Compute the Gaussian envelope
    gaussian_part = np.exp(-(x_prime**2 + gamma**2 * y_prime**2) / (2 * sigma**2))
    
    # Compute the sinusoidal (cosine) component
    cosine_part = np.cos(2 * np.pi * x_prime / lambd + psi)
    
    # Multiply components to get the final kernel
    gabor_kernel = gaussian_part * cosine_part
    
    return gabor_kernel



# Parameters for the Gabor filter
kernel_size = 31      # kernel size (must be odd)
sigma = 5.0           # Gaussian scale
theta_degrees = 45    # orientation in degrees
theta_radians = np.deg2rad(theta_degrees) # convert to radians
lambd = 10.0          # wavelength
gamma = 0.5           # aspect ratio

# Створюємо ядро
manual_gabor_kernel = create_gabor_kernel(kernel_size, sigma, theta_radians, lambd, gamma)

# Visualize the generated kernel to verify it
plt.imshow(manual_gabor_kernel, cmap='gray')
plt.title(f'Ядро Ґабора (ручна генерація), θ={theta_degrees}°')
plt.colorbar()
plt.show()

image_gray = cv2.imread('D:/projects/labs/image_recognition/lab_2/obamna.png', cv2.IMREAD_GRAYSCALE)

if image_gray is None:
    print("Failed to load the image.")
else:
    height, width = image_gray.shape
    # Use float for intermediate calculations
    gabor_filtered_manual = np.zeros_like(image_gray, dtype=np.float32)

    # Offset from borders based on kernel size
    offset = kernel_size // 2

    # Perform convolution
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Extract the region
            region = image_gray[y-offset:y+offset+1, x-offset:x+offset+1]
            
            # Convolution operation
            new_pixel_value = np.sum(region * manual_gabor_kernel)
            
            gabor_filtered_manual[y, x] = new_pixel_value

    # Normalize result to [0,255] for visualization
    gabor_filtered_visual = cv2.normalize(gabor_filtered_manual, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Original grayscale image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(gabor_filtered_visual, cmap='gray')
    plt.title(f'Gabor filter, θ={theta_degrees}°')
    
    plt.show()