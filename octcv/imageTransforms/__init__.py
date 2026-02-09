import sys
sys.path.append('../..')
from octcv.arrViz import *

def low_pass_filter_3D(volume, mask_radius=30):
    # Perform FFT and shift the zero frequency component to the center
    f_transform = np.fft.fftn(volume)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Create a low-pass filter mask
    rows, cols, depth = volume.shape
    crow, ccol, cdepth = rows // 2, cols // 2, depth // 2
    mask_radius = mask_radius # Cutoff frequency: adjust this to control blurring
    
    # Create a circular mask of ones in the center, zeros elsewhere
    low_pass_mask = np.zeros((rows, cols, depth))
    for i in range(rows):
        for j in range(cols):
            for k in range(depth):
                if np.sqrt((i - crow)**2 + (j - ccol)**2 + (k - cdepth)**2) <= mask_radius:
                    low_pass_mask[i, j, k] = 1
    
    # Apply the mask to the frequency domain image
    filtered_f_transform = f_transform_shifted * low_pass_mask
    
    # Perform inverse FFT and take the real component
    f_ishift = np.fft.ifftshift(filtered_f_transform); 
    volume_filtered = np.fft.ifftn(f_ishift); 
    volume_filtered = np.real(volume_filtered) # Take the real part
    return volume_filtered

def low_pass_filter_2D(image, mask_radius=15):
    # Perform FFT and shift the zero frequency component to the center
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Create a low-pass filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask_radius = mask_radius # Cutoff frequency: adjust this to control blurring
    
    # Create a circular mask of ones in the center, zeros elsewhere
    low_pass_mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow)**2 + (j - ccol)**2) <= mask_radius:
                low_pass_mask[i, j] = 1
    
    # Apply the mask to the frequency domain image
    filtered_f_transform = f_transform_shifted * low_pass_mask
    
    # Perform inverse FFT and take the real component
    f_ishift = np.fft.ifftshift(filtered_f_transform)
    image_filtered = np.fft.ifft2(f_ishift)
    image_filtered = np.real(image_filtered) # Take the real part
    
    return image_filtered

