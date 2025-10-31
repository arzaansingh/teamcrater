import os
import cv2 as cv
import dataset as d

dataset_path = "/work/mzhai/lunar_dataset/images"
# get the path to the folder
clean_folder = "/work/mzhai/lunar_dataset/images/clean"
masks_clean_folder = "/work/mzhai/lunar_dataset/masks_clean"

for clean_img in os.listdir(clean_folder):
  clean_img_path = os.path.join(clean_folder, clean_img)

  # read and convert to RGB
  img = cv.imread(clean_img_path, cv.IMREAD_COLOR)
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

  # convert rbg to class label
  mask = d.rgb2Class(img)

  # save the mask as "mask_clean000X.png"
  name = "mask_" + clean_img
  cv.imwrite(os.path.join(masks_clean_folder, name), mask)