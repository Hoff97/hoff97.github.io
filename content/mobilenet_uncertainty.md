Title: Fast uncertainty estimation for mobilenet.
Date: 2020-08-31 16:25
Modified: 2020-08-31 16:25
Category: Machine learning
Tags: deep-learning, uncertainty-estiamtion
Slug: mobilenet-uncertainty
Authors: Frithjof Winkelmann
Summary: Here I explore how uncertainty estimation can be done very quickly with mobilenet
Todos: Add references (Line 32), add example in Detext app

While writing my bachelor thesis, I often ended up using [Detexify](http://detexify.kirelabs.org/classify.html)
to look up latex symbols I forgot. Unfortunately, mobile data doesnt work very
well in germany, so when I was traveling a lot of times I could'nt access this page.
For this reason, I build [Detext](https://detext.haskai.de/client/),
a progressive web app (PWA), that classifies latex symbols without needing
a internet connection. It uses MobileNet [@@Howard2017],
which is run directly on the client side using [onnx.js](https://github.com/microsoft/onnxjs).

![]({filename}images/mobilenet_uncertainty/detext_example.png "Example of detext predicting the latex code for the alpha character")

This works pretty well for most symbols. The only pet peeve I had with
it until recently was, that it wont tell you when its not
certain about the predicted symbol or simply predicts the wrong symbol.
In the following example, the app predicts the $\rightarrow$ (\rightarrow) symbol,
since it does'nt know $\rightharpoonup$ (\rightharpoonup).

![]({filename}images/mobilenet_uncertainty/detext_wrong.png "Detext predicts \rightarrow since it doesnt know \rightharpoonup")

Ideally of course, the prediction would include some uncertainty of the network.
This is known as uncertainty estimation and a variety of approaches exist to
solve it, for example using model ensembles [@@Lakshminarayanan2017], predicting a uncertainty [@@Sequ2019] or
using test time dropout [@@Gal2016].


In this article I will focus on test time dropout. Normally, dropout is
only enabled during training and is used to prevent overfitting.
To enable the model uncertainty, instead of disabling the dropout layers(s)
during testing, one can instead keep it enabled. Feeding in the same input
to a network with test time dropout will then give different outputs,
depending on which neurons stay enabled.

Lets look at an example. The top-5 softmax values that mobilenet predicts
for the following picture
![]({filename}images/mobilenet_uncertainty/rightharpoon.png "Hand drawn right harpoon symbol")

are shown in the plot below:

![]({filename}images/mobilenet_uncertainty/wrong_softmax.png "Softmax of the top-5 classes")

Since the $\rightharpoonup$ symbol is not in the training dataset, these
are all wrong of course. Unfortunately, the softmax score for the top class
is pretty high, so it can not be used as an uncertainty estimate.

To estimate the model uncertainty, we turn on dropout at test time
and look at the model predictions. For this we pass the same image
through the network 100 times and visualize the predictions together with
their variance:

![]({filename}images/mobilenet_uncertainty/rightharpoonerr.png "Softmax of the top-5 classes with error bars")

Here the black bars indicate the standard deviation of the predicted
softmax values. We can see that using test time dropout induces some variance.

If we compare this to the prediction of a symbol that is included in the
dataset, like the following

![Hand drawn right arrow symbol]({filename}images/mobilenet_uncertainty/rightarrow.png "Hand drawn right arrow symbol")

we can see that this has a lower variance:

![Softmax of the top-5 classes with error bars]({filename}images/mobilenet_uncertainty/rightarrowerr.png "Softmax of the top-5 classes with error bars").

Lets look at a few more examples. Here are the variances of symbols that are close to the training set:

![]({filename}images/mobilenet_uncertainty/uncertain_good.png "Variances of images similar to the training set").

And here those of pictures that are not found in the training set:

![]({filename}images/mobilenet_uncertainty/uncertain_bad.png "Variances of images not similar to the training set").

This is nice, since it allows us to detect when the model is uncertain about its
predictions. Unfortunately, this approach requires us to make multiple
forward passes through the model.
In the case of the Detext app this is not acceptable. The app should run on
mobile devices too, and there a forward pass might already take ~1 second.
Doing 10 forward passes just to estimate the uncertainty would take too long.
Fortunately the architecture of mobilenet allows us to estimate the variance
with only one forward pass.

The standard implementation of MobileNet only includes a single dropout layer.
It computes a image feature with the feature network
$f: \mathbb{R}^{W \times H} \rightarrow \mathbb{R}^D$, then applies dropout
and then used a single fully connected layer and a softmax, so in full the prediction
of an image is
$$ pred = softmax(W \times dropout(f(image)) $$

The feature of an image will thus always be the same, even with test time dropout
enabled. We can use a few simple tricks to estimate the variance.

The covariance matrix of $F = dropout(f(image))$ is simply a diagonal matrix
with $\Sigma^F_{ii} = p/(1-p)*f(image)_i^2$ (its just a bernoully random vector
multiplied by $f(image)/(1-p)$ ).

We get the covariance after the linear layer by using the fact that
$X = W F$ has covariance $W \Sigma^F W^T$.
Now we only have to find out how to compute
the variance of $softmax(X)$ given the covariance and mean of $X$.

We can do this using the
[taylor expansion for the moments of random variables](https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables). For the covariance
this reads:

$$ Cov(f(X)) \approx J_f \Sigma^X J_f^T$$

where $J_f$ is the jacobian of $f$ evaluated at $E[X]$.
We only need to find the jacobian of the softmax which is given by:

$$ J^{softmax}_{ij}(X) = \begin{cases}
    softmax(X)_i (1 - softmax(X)_i) & i=j\\
    -softmax(X)_j softmax(X)_i & i \neq j
\end{cases} $$

In total the evaluation of the variance is done by

$$
F = f(image)\\
\Sigma^F_{ii} = \tfrac{p}{1-p}*F_i^2\\
X = W F\\
\Sigma^X = W \Sigma^F W^T\\
S = softmax(X)\\
\Sigma^S = J^{softmax}(X) \Sigma^X J^{softmax}(X)^T
$$

Lets compare the variance we get with this estimate to the variance estimated from sampling:

![]({filename}images/mobilenet_uncertainty/variance_comparison.png "Comparison of variance estimated from sampling and approximation")

We can see that the approximated variance is not exactly equal to the variance estimated from sampling, but it is generally
higher for images that are not similar from the training set.

This can be used as a rough indicator, when the prediction of MobileNet might not be trusted.
