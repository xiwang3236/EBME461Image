# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider

# amp = 1
# freq = 1

# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.25)
# t = np.arange(0.0, 1.0, 0.001)
# s = amp * np.sin(2*np.pi*freq*t)
# l, = plt.plot(t, s, lw=2)

# ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# slider = Slider(ax_slider, 'Freq', 0.1, 30.0, valinit=freq, valstep=1)

# def update(val):
#     freq = slider.val
#     l.set_ydata(amp*np.sin(2*np.pi*freq*t))
#     fig.canvas.draw_idle()

# slider.on_changed(update)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import cv2

def apply_band_reject(f_transform, center_x, center_y, width):
    # Create a mask for the band reject filter
    rows, cols = f_transform.shape
    y, x = np.ogrid[-rows//2:rows//2, -cols//2:cols//2]
    
    # Calculate distance from center point
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create the mask (1 = pass, 0 = reject)
    mask = np.ones((rows, cols))
    mask[dist <= width] = 0
    
    # Apply symmetrically (important for Fourier transform)
    center_x_sym, center_y_sym = -center_x, -center_y
    dist_sym = np.sqrt((x - center_x_sym)**2 + (y - center_y_sym)**2)
    mask[dist_sym <= width] = 0
    
    return f_transform * mask

def setup_interactive_filter():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)
    
    # Load image
    img = cv2.imread(r'D:\drive\OneDrive - Case Western Reserve University\FILE\2025spring\EBME461 Image\GroupProject\HW05\HW05_EBMECSDS_361461_Images\FigP0405(HeadCT_corrupted).tif', 0)  # 0 for grayscale

    # Compute the 2D FFT
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    
    # Display original image
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Display spectrum
    spectrum_display = ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.set_title('Frequency Spectrum')
    ax2.axis('off')
    
    # Prepare filtered image display
    filtered_display = ax3.imshow(img, cmap='gray')
    ax3.set_title('Filtered Result')
    ax3.axis('off')
    
    # Create slider for filter width
    ax_width = plt.axes([0.2, 0.1, 0.6, 0.03])
    width_slider = Slider(ax_width, 'Filter Width', 1, 50, valinit=10)
    
    # Store the current filter center
    filter_center = {'x': None, 'y': None}
    
    def on_click(event):
        if event.inaxes == ax2:
            filter_center['x'] = int(event.xdata - magnitude_spectrum.shape[1]//2)
            filter_center['y'] = int(event.ydata - magnitude_spectrum.shape[0]//2)
            update(width_slider.val)
    
    def update(val):
        if filter_center['x'] is not None and filter_center['y'] is not None:
            # Apply filter
            filtered_f = apply_band_reject(f_shift, 
                                         filter_center['x'], 
                                         filter_center['y'], 
                                         width_slider.val)
            
            # Inverse FFT
            filtered_img = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_f)))
            
            # Update displays
            filtered_display.set_data(filtered_img)
            fig.canvas.draw_idle()
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_click)
    width_slider.on_changed(update)
    
    plt.show()

# Run the interactive filter
setup_interactive_filter()