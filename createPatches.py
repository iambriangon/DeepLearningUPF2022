import sys
import cv2
import os
from tqdm import tqdm
from patchify import patchify

path = sys.argv[1]
new_path = sys.argv[2]


def createPatch(path, new_path, img_name):
    img = cv2.imread(path)
    patches_img = patchify(img, (700, 700, 3), step=700)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, 0, :, :, :]
            if not cv2.imwrite(new_path + '/' + img_name + '_' + str(i).zfill(2) + '_' + str(j).zfill(2) + '.png',
                               single_patch_img):  # Save as PNG, not JPEG for keeping the quality.
                raise Exception("Could not write the image")


if __name__ == '__main__':
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for subdir, dirs, files in os.walk(path):
        for image in tqdm(files):
            img_name = os.path.splitext(image)[0]
            name = os.path.join(subdir, image)
            createPatch(name, new_path, img_name)
