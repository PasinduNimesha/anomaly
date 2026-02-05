import os
import cv2
import numpy as np
import glob

def matlab_style_gauss2d(shape, sigma):
    """
    Approximation of MATLAB's fspecial('gaussian', shape, sigma)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def run_gwfs():
    dataset_names = ["sixray"]
    
    # Target image used for stylization (Ensure this file exists in the directory)
    target_img_name = 'P00002.jpg'
    
    if not os.path.exists(target_img_name):
        print(f"Warning: Stylization source image '{target_img_name}' not found.")
        return

    for d_name in dataset_names:
        print(f"Processing dataset: {d_name}")
        
        # Construct paths
        # Assuming structure: datasets/dataset_name/input/
        input_dir = os.path.join('datasets', d_name, 'input')
        output_dir = os.path.join('datasets', d_name, 'abnormal')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get all png images
        search_path = os.path.join(input_dir, '*.png')
        files = glob.glob(search_path)
        
        # Load and prep target image 't'
        t = cv2.imread(target_img_name)
        if t is None:
            continue
            
        # Ensure 3 channels
        if len(t.shape) == 2:
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        
        # Resize t to 2240x2240
        t = cv2.resize(t, (2240, 2240), interpolation=cv2.INTER_LINEAR)
        
        # FFT of target 't'
        # We need to process per channel or treat as 3D. 
        # MATLAB's fft2 on a 3D array computes fft2 on each page (channel) independently.
        # Numpy fft2 needs axes specified if 3D.
        ac = np.fft.fft2(t, axes=(0, 1))
        ac2 = np.fft.fftshift(ac, axes=(0, 1))
        acR = np.abs(ac2)
        # acI = np.angle(ac2) # Unused in the mixing logic logic below? 
        # Actually logic uses 'acR' for mixing.
        
        for file_path in files:
            fn = os.path.basename(file_path)
            save_path = os.path.join(output_dir, fn)
            
            # Read source image 's'
            s = cv2.imread(file_path)
            if s is None:
                continue
                
            if len(s.shape) == 2:
                s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
                
            # Resize s to 2240x2240
            s_resized = cv2.resize(s, (2240, 2240), interpolation=cv2.INTER_LINEAR)
            
            # Use dimensions of 't' (which is 2240x2240) for 's' as well to match logic
            r, c, ch = t.shape
            
            # Resize s to match t (redundant here as both are 2240, but keeping logic)
            s_final = cv2.resize(s_resized, (c, r), interpolation=cv2.INTER_LINEAR)
            
            # FFT of source
            ab = np.fft.fft2(s_final, axes=(0, 1))
            ab1 = np.fft.fftshift(ab, axes=(0, 1))
            abR = np.abs(ab1)
            abI = np.angle(ab1)
            
            # Gaussian Window
            r1, c1, ch1 = s_final.shape
            window_size = (r1, c1)
            si = 1
            
            # Create Gaussian filter
            G = matlab_style_gauss2d(window_size, si)
            
            # Normalize G to 0-1 range based on min/max
            g_min = np.min(G)
            g_max = np.max(G)
            if g_max - g_min != 0:
                G = (G - g_min) / (g_max - g_min)
            else:
                G = np.zeros_like(G)
                
            # Expand G to match channels for broadcasting
            G = G[:, :, np.newaxis]
            
            # Mix Amplitudes: S = acR .* G
            S = acR * G
            
            # Add to source amplitude
            abR_new = abR + S
            
            # Reconstruct
            # ab2 = abR .* exp(abI * 1i)
            ab2 = abR_new * np.exp(1j * abI)
            
            ab2 = np.fft.ifftshift(ab2, axes=(0, 1))
            ab2 = np.fft.ifft2(ab2, axes=(0, 1))
            ab2 = np.real(ab2)
            
            # Mat2Gray equivalent: (val - min) / (max - min)
            # This is typically done globally or per channel? MATLAB's mat2gray is global if input is 3D?
            # Actually mat2gray scales values to [0,1].
            ab2_min = np.min(ab2)
            ab2_max = np.max(ab2)
            if ab2_max - ab2_min != 0:
                ab2 = (ab2 - ab2_min) / (ab2_max - ab2_min)
            else:
                ab2 = np.clip(ab2, 0, 1)

            # Display would go here (cv2.imshow), skipping for batch script
            
            # Resize back to 2240x2240 (Already there, but following logic)
            ab2 = cv2.resize(ab2, (2240, 2240), interpolation=cv2.INTER_LINEAR)
            
            # Save (Convert to 0-255 uint8)
            ab2_uint8 = (ab2 * 255).astype(np.uint8)
            
            # Determine extension from user code (logic uses '.PNG' hardcoded in imwrite line)
            # But the file might have been .png original. 
            cv2.imwrite(save_path, ab2_uint8)

if __name__ == "__main__":
    run_gwfs()