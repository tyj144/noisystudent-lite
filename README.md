# noisystudent-lite
NoisyStudent Lite. Final project for CSCI 2952-C.


## TODOs
Fix the pseudolabeling (teacher is spitting out only 46 or 47), which seems like
it's not reading in model


Figure out why it trains well with a batch size of 16 but not 8
Figure out how to mix image sets
- Train on the mixed imaged set
- Or, train separately on labeled and pseudolabeled set
Add input noise per paper via RandAugment
Add model noise per paper
Train student (or teacher too) as a larger model (like ResNet-512)

Potential fixes for pseudolabeling:
- Balance the dataset
- Balance the batch sizes

Q: Do I use mean and stddev from ImageNet or from CUB 200?

## Extra time:
Implement soft pseudo-labeling
Baseline: train on both labeled sets together with a bigger model as a baseline


DONE Abstract out dependencies between CUB200.py and pseudolabel.py
 
DONE Train the small hymenoptera model on CCV or GCP as a test run.
DONE Then, move the training of CUB onto it. Evaluate it on the test
set and save it. Make sure we can use it as a "teacher" model.

DONE 1. Find a way to split train/val in CUB
DONE 2. Find a way to store pseudo labels
DONE 3. Train on CCV













Brainstorming from earlier
--------------------------------------------------------
1. Train the initial ResNet-50 or DenseNet-121 on CUB 200 2010 (smaller) set with labels.
2. Use this model to pseudo-label the larger CUB 200 2011 set.
    - Q: How do I store the pseudo labels? Do I need to label all at once then train?
    A: Pseudolabel help: https://towardsdatascience.com/pseudo-labeling-to-deal-with-small-datasets-what-why-how-fd6f903213af
3. Train the student model on labeled from CUB 200 and then CUB 200 2011.
    - Q: How do I get the custom loss from the paper? How long might this take?

... In between, iterate if possible.
4. See if performance on original CUB 200 dataset improves. Evaluate performance.

Warning about overlap is probably not a big deal: https://guopei.github.io/2016/Overlap-Between-Imagenet-And-CUB/

One epoch of Tiny ImageNet through all 100k images takes ~7hrs on a CPU.


--------------------------------------------------------

Criteria for success: If I can replicate the increase in robustness with NoisyStudent
compared to a vanilla model on a smaller scale, that shows the NoisyStudent method
itself is what's leading to better performance.

Regular? set: Tiny ImageNet (200 classes, 10k images)
    - Original: ImageNet (10M images, 10k+ classes)
Unlabeled set: ImageNet 1%? (7 GB) or 10% (19GB)
    - https://www.tensorflow.org/datasets/catalog/imagenet2012_subset
    - Original: JFT-300M (300M images, 81M after de-dupe), i.e. 30x more images - 8x more images after de-dupe

Apparently ResNet gets a straight up zero on a lot of the ImageNet-A examples.

I can either:

1. Use a pre-trained ResNet/DenseNet, etc. with the ImageNet weights
    - Then, use it as a teacher model to pseudo-label new images.
    - PRO(s): Easier because I don't have to train
    - CON(s): I don't have a larger dataset to train on.
2. Train a ResNet/DenseNet from scratch with Tiny ImageNet, then pseudolabel the larger ImageNet?
    - PRO(s): More accurate to the scale (will probably show a bigger improvement compared to baseline)
    - CON(s): Will take longer to train


NoisyStudent
- Train a big EfficientNet on the ImageNet dataset. This will be the fully-supervised teacher model.
- Take the teacher and train it on unlabeled images from JFT-300M. 
    - Only take the ones that have confidence > 0.3. (Select at most 130K images.)
    - They also duplicate images to balance the set across classes (each one has 130k).
    - Unlabeled data is deliberately **noised** with RandAugment. 
      Student model is also noised with dropout and stochastic depth.
- Train the student with both the labeled and the pseudo-labeled images. (they choose soft)
- Student becomes the teacher, iterate.

Set up `~/tensorflow_datasets/downloads/manual/` via https://www.tensorflow.org/datasets/catalog/imagenet2012_subset





