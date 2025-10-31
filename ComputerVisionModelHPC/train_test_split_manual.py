import os

# Train test split manually
def train_test_split_manual(test_size, val_size):
  import shutil

  image_source_folder = "/work/mzhai/lunar_dataset/images/render"
  mask_source_folder = "/work/mzhai/lunar_dataset/masks_clean"

  list_of_images = sorted(os.listdir(image_source_folder))
  list_of_masks = sorted(os.listdir(mask_source_folder))

  assert(len(list_of_images) == (len(list_of_masks))-3)

  num_samples = len(list_of_images)
  test_i = round(num_samples*test_size)
  val_i = round(num_samples*test_size)

  image_train_folder = "/work/mzhai/lunar_dataset/images/train"
  # os.mkdir(image_train_folder)
  image_val_folder = "/work/mzhai/lunar_dataset/images/validation"
  # os.mkdir(image_val_folder)
  image_test_folder = "/work/mzhai/lunar_dataset/images/test"
  # os.mkdir(image_test_folder)

  mask_train_folder = "/work/mzhai/lunar_dataset/masks_clean/train"
  # os.mkdir(mask_train_folder)
  mask_val_folder = "/work/mzhai/lunar_dataset/masks_clean/validation"
  # os.mkdir(mask_val_folder)
  mask_test_folder = "/work/mzhai/lunar_dataset/masks_clean/test"
  # os.mkdir(mask_test_folder)

  # copy the test images first
  for i in range(test_i):
    shutil.copy(os.path.join(image_source_folder, list_of_images[i]), os.path.join(image_test_folder, list_of_images[i]))
    shutil.copy(os.path.join(mask_source_folder, list_of_masks[i]), os.path.join(mask_test_folder, list_of_images[i]))

  for i in range(test_i, test_i+val_i):
    shutil.copy(os.path.join(image_source_folder, list_of_images[i]), os.path.join(image_val_folder, list_of_images[i]))
    shutil.copy(os.path.join(mask_source_folder, list_of_masks[i]), os.path.join(mask_val_folder, list_of_images[i]))

  for i in range(test_i+val_i, num_samples):
    shutil.copy(os.path.join(image_source_folder, list_of_images[i]), os.path.join(image_train_folder, list_of_images[i]))
    shutil.copy(os.path.join(mask_source_folder, list_of_masks[i]), os.path.join(mask_train_folder, list_of_images[i]))

train_test_split_manual(0.2, 0.2)