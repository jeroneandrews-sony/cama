# Conditional Adversarial Camera Model Anonymization

PyTorch implementation of [Conditional Adversarial Camera Model Anonymization](https://arxiv.org/abs/2002.07798) (ECCV 2020 Advances in Image Manipulation Workshop).

The paper was selected for an oral presentation at the ECCV 2020 Advances in Image Manipulation Workshop.

The model of camera that was used to capture a particular photographic image (model attribution) is typically inferred from high-frequency model-specific artifacts present within the image. Model anonymization is the process of transforming these artifacts such that the apparent capture model is changed.

The method proposed, Conditional Adversarial Camera Model Anonymization (Cama), offers a way to preserve privacy by transforming an image's ground truth camera model-specific artifacts to those of a disparate target camera model.

Cama is able to reliably perform targeted transformations. Importantly, not only can Cama successfully perform targeted transformations on in-distribution images (i.e. captured by camera models known to it), but also on out-of-distribution images (i.e. captured by camera models unknown to it).

![inDist-example](images/flower.png)

![outDist-example](images/building.png)

