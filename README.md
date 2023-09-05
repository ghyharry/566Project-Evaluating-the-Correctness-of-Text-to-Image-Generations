# Evaluating the Correctness of Text-to-Image Generations
One Eye Dingzhen

Our methods has two approaches, they will evaluate physical consistency score of an image. CLIP-Physical Rules has been trained using CLIP pretrained which is published by OpenAI, and fine-tuned it. Segment+ViT is using vit-base-patch16-224-in21k pretrained model on ImageNet-21k.

## Usage
Generated Data
- Run __resize.py__ to resize, shift, rotate, and flip image for data augmentation.

- __main.py__, in our early stage, we used CNN to test on hand images.

Fine-tuning CLIP
- Run __Clip_finetune.ipynb__ to fine-tune CLIP, you may apply on your new data, with correct free-form captions. Google Colab environment is recommended. It will produce a physical consistency score.

Segment+ViT
- Run [CDCL human part segmentation](https://github.com/kevinlin311tw/CDCL-human-part-segmentation/tree/master) to generated highlighted human body parts in pictures.
Note: if __cdcl_environment.yaml__ doesn't work, you will need to install package manually.
- Run __vit_ipynb.ipynb__ to import segmented pictures to the pretraiend model, and fine-tuning ViT model introduced above.
