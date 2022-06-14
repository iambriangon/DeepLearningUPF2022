import sys
import cv2
import os
import argparse
from tqdm import tqdm
from patchify import patchify


def createPatch(img_path, data_path, img_name, w, h, s):
    img = cv2.imread(img_path)
    patches_img = patchify(img, (w, h, 3), step=s)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, 0, :, :, :]
            if not cv2.imwrite(data_path + '/' + img_name + '_' + str(i).zfill(2) + '_' + str(j).zfill(2) + '.png',
                               single_patch_img):  # Save as PNG, not JPEG for keeping the quality.
                raise Exception("Could not write the image")


def organize(path):
    os.system("cd {}".format(path))
    os.system("mkdir test/ test/Brayan test/Cinta test/Pol train/ train/Brayan train/Cinta train/Pol")
    os.system("mv Brayan_1* Brayan_2* train/Brayan")
    os.system("mv Cinta_1* Cinta_2* train/Cinta")
    os.system("mv Pol_1* Pol_2* train/Pol")
    os.system("mv Brayan_3* test/Brayan")
    os.system("mv Cinta_3* test/Cinta")
    os.system("mv Pol_3* test/Pol")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-path', default='imgs/crop', dest='img_path', type=str,
                        help="Path containing a folder with your images to patch")
    parser.add_argument('--data-path', default='data', dest='data_path', type=str,
                        help="Path for storing the patches")
    parser.add_argument('--w', default=500, dest='width', type=int,
                        help="Width patch")
    parser.add_argument('--h', default=500, dest='height', type=int,
                        help="Height patch")
    parser.add_argument('--s', default=500, dest='stride', type=int,
                        help="Stride patch")

    return parser


def main(params):
    if os.path.exists(params.data_path):
        # Delete directory if already had other data and create new one
        os.system("rm -r {}".format(params.data_path))
        os.system("mkdir {}".format(params.data_path))
    else:
        os.system("mkdir {}".format(params.data_path))

    for subdir, dirs, files in os.walk(params.img_path):
        for image in tqdm(files):
            img_name = os.path.splitext(image)[0]
            img_path = os.path.join(subdir, image)
            createPatch(img_path, params.data_path, img_name,
                        params.width, params.height, params.stride)

    organize(params.data_path)


if __name__ == '__main__':
    main(get_args().parse_args())
