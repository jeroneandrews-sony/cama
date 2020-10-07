# Conditional Adversarial Camera Model Anonymization

PyTorch implementation of [Conditional Adversarial Camera Model Anonymization](https://arxiv.org/abs/2002.07798) (ECCV 2020 Advances in Image Manipulation Workshop).

Digital photographs can be **blindly** attributed to the specific camera model used for capture.

![blind-att](images/blind-att.png)

Conditional Adversarial Camera Model Anonymization (Cama) offers a way to preserve privacy by transforming these artifacts such that the apparent capture model is changed (targeted transformation). That is, given an image and a target label condition, the applied transformation causes a non-interactive black-box *target* (i.e. to be attacked/fooled) convnet classifier <img src="https://render.githubusercontent.com/render/math?math=\large F"> to predict the target label given the transformed image. While at the same time retaining the original image content.

However, Cama is trained in a **non-interactive black-box setting**: Cama does not have knowledge of the parameters, architecture or training randomness of <img src="https://render.githubusercontent.com/render/math?math=\large F">, nor can Cama interact with it. 

![blind-att](images/cam-anon.png)

## Attack Setting and Desiderata
We denote by <img src="https://render.githubusercontent.com/render/math?math=\large x\in\mathbb{R}^d"> and <img src="https://render.githubusercontent.com/render/math?math=\large y\in\mathbb{N}_c=\{1,\dots,c\}"> an image and its ground truth (source) camera model label, respectively, sampled from a dataset <img src="https://render.githubusercontent.com/render/math?math=\large p_{\text{data}}">. Consider a \emph{target} (i.e. to be attacked) convnet classifier <img src="https://render.githubusercontent.com/render/math?math=\large F"> with <img src="https://render.githubusercontent.com/render/math?math=\large c"> classes trained over input-output tuples <img src="https://render.githubusercontent.com/render/math?math=\large (x,y)\sim p_{\mathrm{data}}(x,y)">. Given <img src="https://render.githubusercontent.com/render/math?math=\large x">, <img src="https://render.githubusercontent.com/render/math?math=\large F"> outputs a prediction vector of class probabilities <img src="https://render.githubusercontent.com/render/math?math=\large {F:x\mapsto F(x)\in[0,1]^{c}}">.

In this work, we operate in a **non-interactive black-box setting**: we do not assume to have knowledge of the parameters, architecture or training randomness of <img src="https://render.githubusercontent.com/render/math?math=\large F">, nor can we interact with it. We do, however, assume that we can sample from a dataset similar to <img src="https://render.githubusercontent.com/render/math?math=\large p_{\mathrm{data}}">, which we denote by <img src="https://render.githubusercontent.com/render/math?math=\large q_{\mathrm{data}}">. Precisely, we can sample tuples of the following form: <img src="https://render.githubusercontent.com/render/math?math=\large (x,y)\sim q_{\text{data}}(x,y)"> s.t. <img src="https://render.githubusercontent.com/render/math?math=\large y\in\mathbb{N}_{c^\prime}">, where <img src="https://render.githubusercontent.com/render/math?math=\large (x,y)\sim q_{\text{data}}(x,y)"> s.t. <img src="https://render.githubusercontent.com/render/math?math=\large c^\prime \leq c">. That is, the set of possible image class labels in <img src="https://render.githubusercontent.com/render/math?math=\large p_{\text{data}}"> is a superset of the set of possible image class labels in <img src="https://render.githubusercontent.com/render/math?math=\large Q_{\text{data}}">, i.e. <img src="https://render.githubusercontent.com/render/math?math=\large \mathbb{N}_{c}\supseteq \mathbb{N}_{c^\prime}">.

<!-- Suppose $(x,y)\sim q_{\text{data}}(x,y)$ and $y^\prime \in\mathbb{N}_{c^\prime}$, where $y^\prime \neq y$ is a target label. Our aim is to learn a function $G:(x,y^\prime)\mapsto x^\prime \approx x$ s.t. the maximum probability satisfies $\argmax_{i} F(x^\prime)_i=y^\prime$. This is known as a \emph{targeted} attack, whereas the maximum probability of an \emph{untargeted} attack must satisfy $\arg \max_{i} F(x^\prime)_i\neq y$. This work focuses on targeted attacks. -->


<!-- Significantly, the applied transformations do not alter an image's content and are (largely) imperceptible.


That is, a *target* (i.e. to be attacked) convnet classifier F

That is, we wish to learn how to transform the artefacts of images based on target camera model label conditions such that the classifier F outputs the target label. While at the same time retaining the image content.

However, we operate in a non-interactive black-box setting: we do not assume to have knowledge of the parameters, architecture or training randomness of F, nor can we interact with it.  -->




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
