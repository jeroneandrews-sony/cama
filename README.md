# Conditional Adversarial Camera Model Anonymization

PyTorch implementation of [Conditional Adversarial Camera Model Anonymization](https://arxiv.org/abs/2002.07798) (ECCV 2020 Advances in Image Manipulation Workshop).

The model of camera that was used to capture a particular photographic image (model attribution) is typically inferred from high-frequency model-specific artifacts present within the image. Conditional Adversarial Camera Model Anonymization (Cama) offers a way to preserve privacy by transforming these artifacts such that the apparent capture model is changed (targeted misclassification). 

## Anonymizing in-distribution images
Cama is able to successfully perform targeted transformations on in-distribution images (i.e. images captured by camera models known to it).

![inDist-example](images/flower.png)

## Anonymizing out-of-distribution images
Cama is also able to successfully perform targeted transformations on out-of-distribution images (i.e. images captured by camera models unknown to it).

![outDist-example](images/building.png)

