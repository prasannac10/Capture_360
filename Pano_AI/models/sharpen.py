import cv2

def unsharp_mask(image, amount=0.35, sigma=1.0):
    blur=cv2.GaussianBlur(image,(0,0),sigma)
    return cv2.addWeighted(image,1.0+amount,blur,-amount,0)
