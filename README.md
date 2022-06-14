# DeepLearningUPF2022

## Description
This project was made by Brayan González, Pol Ayala, Cinta Arnau.

The goal for this project is to train a NN so it can classify an image of handwritten text made by any of us.

We tried a hand made CNN, and includes some famous architectures such as ResNet, MobileNet, AlexNet, VGG (pretrained or not)
## Usage
### Create patches from handwritten images
Creates a training/test dataset with patches of size WxH (removes last dataset if DATA PATH already exists and contained data!!!)

**usage**: createPatches.py [-h] [--img-path IMG_PATH] [--data-path DATA_PATH] [--w WIDTH] [--h HEIGHT] [--s STRIDE]

### CNN Classification
**usage**: python 3 classify.py [-h] [--train-path TRAIN_PATH] [--test-path TEST_PATH] [--result-path RESULTS_PATH] [--lr LR] [--batch-size BATCH_SIZE]
                   [--model {CNN,AlexNet,ResNet,MobileNet,VGG}] [--pretrained PRETRAINED(flag)] [--epochs EPOCHS] [--num-class N_CLASS]

for help type: python3 -h