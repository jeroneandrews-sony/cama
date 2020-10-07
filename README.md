# Conditional Adversarial Camera Model Anonymization

PyTorch implementation of [Conditional Adversarial Camera Model Anonymization](https://arxiv.org/abs/2002.07798) (ECCV 2020 Advances in Image Manipulation Workshop).

Digital photographs can be blindly attributed to the specific camera model used for capture.

![blind-att](images/blind-att.png)

Conditional Adversarial Camera Model Anonymization (Cama) offers a way to preserve privacy by transforming these artifacts such that the apparent capture model is changed (targeted transformation). Significantly, the applied transformations do not alter an image's content and are (largely) imperceptible.

## Anonymizing in-distribution images
Cama is able to successfully perform targeted transformations on in-distribution images (i.e. images captured by camera models known to it).

Example (below) of Cama transformed images <img src="https://render.githubusercontent.com/render/math?math=\large x^\prime"> with different target label conditions <img src="https://render.githubusercontent.com/render/math?math=\large y^\prime"> given an in-distribution input image <img src="https://render.githubusercontent.com/render/math?math=\large x"> (whose ground truth label is <img src="https://render.githubusercontent.com/render/math?math=\large y">). The applied transformations (amplified for visualization purposes) are shown as <img src="https://render.githubusercontent.com/render/math?math=\large \delta">.

![inDist-example](images/flower.png)



## Anonymizing out-of-distribution images
Cama is also able to successfully perform targeted transformations on out-of-distribution images (i.e. images captured by camera models unknown to it).

Example (below) of Cama transformed images <img src="https://render.githubusercontent.com/render/math?math=\large x^\prime"> with different target label conditions <img src="https://render.githubusercontent.com/render/math?math=\large y^\prime"> given an out-of-distribution input image <img src="https://render.githubusercontent.com/render/math?math=\large x"> (whose ground truth label is <img src="https://render.githubusercontent.com/render/math?math=\large y">). The applied transformations (amplified for visualization purposes) are shown as <img src="https://render.githubusercontent.com/render/math?math=\large \delta">.

![outDist-example](images/building.png)

## Model
![cama-model](images/model.png)

Cama has two class conditional components: a generator G that transforms an image x conditioned on a target class label y′, and a discriminator D that predicts whether the low-level high-frequency pixel value dependency features of any given image conditioned on a label are real or fake. In addition, Cama has a fixed (w.r.t. its parameters) dual-stream discriminative decision-making component E (evaluator) that decides whether a transformed image x belongs to its target class y . In essence, E serves as a surrogate for the non-interactive black-box F. W.r.t. E, a transformed image x′ is decomposed into its high and low spatial frequency components (x′H and x′L, respectively), via E0, with each assigned to a separate stream (EH and EL, respectively). The evaluator then reasons over the information present in x′H and x′L separately (via EH and EL, respectively). This reinforces the transformation process, as G is constrained
