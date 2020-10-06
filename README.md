# Conditional Adversarial Camera Model Anonymization

PyTorch implementation of [Conditional Adversarial Camera Model Anonymization](https://arxiv.org/abs/2002.07798) (ECCV 2020 Advances in Image Manipulation Workshop).

The model of camera that was used to capture a particular photographic image (model attribution) is typically inferred from high-frequency model-specific artifacts present within the viewable image (i.e. a .jpeg or .tiff image). Conditional Adversarial Camera Model Anonymization (Cama) offers a way to preserve privacy by transforming these artifacts such that the apparent capture model is changed (targeted transformation). Significantly, the applied transformations do not alter an image's content and are (largely) imperceptible.

## Anonymizing in-distribution images
Cama is able to successfully perform targeted transformations on in-distribution images (i.e. images captured by camera models known to it).

Example (below) of Cama transformed images <img src="https://render.githubusercontent.com/render/math?math=\large x^\prime"> with different target label conditions <img src="https://render.githubusercontent.com/render/math?math=\large y^\prime"> given an in-distribution input image <img src="https://render.githubusercontent.com/render/math?math=\large x"> (whose ground truth label is <img src="https://render.githubusercontent.com/render/math?math=\large y \neq y^\prime">). The applied transformations (amplified for visualization purposes) are shown as <img src="https://render.githubusercontent.com/render/math?math=\large \delta">.
![inDist-example](images/flower.png)



## Anonymizing out-of-distribution images
Cama is also able to successfully perform targeted transformations on out-of-distribution images (i.e. images captured by camera models unknown to it).

![outDist-example](images/building.png)

## Model
