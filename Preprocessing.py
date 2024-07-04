import cv2
import os
import numpy as np

#Define the path to the image
image_path = '/home/data/images/0.png'

# Load the image using OpenCV
image = cv2.imread(image_path)

if image is not None:
    # Print the shape of the image
    print("Image shape:", image.shape)
else:
    print("Error: Image not found or unable to load.")

def preprocess_retinal_image(image_path):
    """
    Preprocesses a retinal image and its corresponding mask for vessel segmentation.

    Parameters:
    image_path (str): Path to the input retinal image.

    Returns:
    preprocessed_image (numpy.ndarray): The preprocessed retinal images
    """
    image = cv2.imread(os.path.join(image_path,i))
    #mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Step 2: Separate channels (BGR)
    b, g, r = cv2.split(image)

    # Step 3: Histogram Equalization for each channel
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # Step 4: Merge channels back together
    equalized_image = cv2.merge([b_eq, g_eq, r_eq])

      # Step 5: Resize the image to (256, 256)
    #resized_image = cv2.resize(gray_image, (256, 256))

    # Step 6: Histogram Equalization
    #equalized_image = cv2.equalizeHist(image)

    # Step 7: Denoising using Gaussian filter
    denoised_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    # Step 8: Morphological operations
    # Use a kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)

    # Top-hat transformation
    tophat = cv2.morphologyEx(denoised_image, cv2.MORPH_TOPHAT, kernel)

    # Step 9: Normalization
    normalized_image = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)    

    return normalized_image


def preprocess_retinal_mask(mask_path):
    """
    Preprocesses a retinal image and its corresponding mask for vessel segmentation.

    Parameters:
    image_path (str): Path to the input retinal image.

    Returns:
    preprocessed_image (numpy.ndarray): The preprocessed retinal images
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(mask.shape)

    #Binarize the mask
    _, binary_mask  = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return binary_mask


images_path = '/home/data/images'
masks_path = '/home/data/labels'

for i in os.listdir(images_path):
  img_num = i.split(".")[0]
  image_path = os.path.join(images_path,i)
  print(image_path)
  preprocessed_image = preprocess_retinal_image(images_path)
  file = "/home/data/processed_images/{img_num}.npy".format(img_num=img_num)
  np.save(file,preprocessed_image)
  print(preprocessed_image)
  # If you want to visualize the results using OpenCV
  '''
  cv2.imshow("Preprocessed Image", preprocessed_image)
  #cv2.imshow("Preprocessed Mask", preprocessed_mask)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  '''

masks_path = '/home/data/labels'
for i in os.listdir(masks_path):
  mask_num = i.split(".")[0]
  mask_path = os.path.join(masks_path,i)
  print(mask_path)
  binary_mask1 = preprocess_retinal_mask(mask_path)
  file = "/home/data/processed_labels/{mask_num}.npy".format(mask_num=mask_num)
  np.save(file,binary_mask1)
  print(binary_mask1)

print(binary_mask1.shape)

print(preprocessed_image.shape)
