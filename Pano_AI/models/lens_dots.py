import cv2
import numpy as np

def remove_lens_dots(image, threshold=245, radius=3):
    """Classical bright-dot detector + Telea inpainting; conservative by design."""
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) if image.ndim==3 else image
    mask=(gray>=threshold).astype(np.uint8)*255
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)); mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    return cv2.inpaint(image,mask,radius,cv2.INPAINT_TELEA)
