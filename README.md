
# mixupy

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Code Quality][quality-image]][quality-url]

mixupy is a python package for data-augmentation inspired by
[mixup: Beyond Empinical Risk Minimization](https://arxiv.org/abs/1710.09412)

If you like mixupy, give it a star, or fork it and contribute!


## Usage

Create additional training data for the iris dataset:
```python
import numpy as np
import pandas as pd
from mixupy import mixup

# Use 'iris' dataset from seaborn package
import seaborn as sns
iris = sns.load_dataset('iris')

# One-hot encode species column
iris_df = pd.get_dummies(iris, columns=['species'], prefix='', prefix_sep='')
iris_df

# Strictly speaking this is 'input mixup' (see Details section below)
set.seed(42)
iris_mix = mixup(iris_df)
iris_mix.describe()
iris_df.describe()

# Further info
help(mixup)
```


## Installation

```python
pip install mixupy
```

Requires python 3.7 or higher plus pandas and numpy

```python
pip install numpy pandas
```


## Details

The mixup function enlarges training sets using linear interpolations
of features and associated labels as described in
[https://arxiv.org/abs/1710.09412](https://arxiv.org/abs/1710.09412).

Virtual feature-target pairs are produced from randomly drawn
feature-target pairs in the training data.
The method is straight-forward and data-agnostic.  It should
result in a reduction of generalisation error.

mixup constructs additional training examples:

x' = λ * x_i + (1 - λ) * x_j, where x_i, x_j are raw input vectors

y' = λ * y_i + (1 - λ) * y_j, where y_i, y_j are one-hot label encodings

(x_i, y_i) and (x_j ,y_j) are two examples drawn at random from the
training data, and λ ∈ [0, 1] with λ ∼ Beta(α, α) for α ∈ (0, ∞).
The mixup hyper-parameter α controls the strength of interpolation between
feature-target pairs.

### mixup() parameters

| Parameter  | Description                                         | Type              | Notes                                 |
|------------|-----------------------------------------------------|-------------------|---------------------------------------|
| data       | Original data                                       | pandas data frame | Required parameter                    |
| alpha      | Hyperparameter specifying strength of interpolation | numeric           | Defaults to 4                         |
| concat     | Concatenate mixup data with original data           | boolean           | Defaults to False                     |
| batch_size | How many mixup values to produce                    | integer           | Defaults to number of 'data' examples |

The 'data' parameter must be a numeric (integers and/or floats) pandas
data frame.  Non-finite values are not permitted.  Categorical variables
should be one-hot encoded.

Alpha values must be greater than or equal to zero.  Alpha equal to zero
specifies no interpolation.

The mixup function returns a pandas data frame containing interpolated
values.  Optionally, the original values can be concatenated with the
new values using the `concat = True` option.

### Mixup with deep learning versus other learning methods

It is worthwhile distinguishing between mixup usage with
deep learning and other learning methods.  Mixup with deep learning
can improve generalisation when a new mixed dataset is generated
every epoch or even better for every minibatch.  This level
of granularity may not be possible with other learning
methods.  For example, simple linear modeling may not
benefit much from training on a single (potentially greatly
expanded) pre-mixed dataset.  This single pre-mixed dataset
approach is sometimes referred to as 'input mixup'.

In certain situations, under-fitting can occur when conflicts
between synthetic labels of the mixup examples and
labels of the original training data are present.  Some learning
methods may be more prone to this under-fitting than others.

### Data augmentation as regularisation

Data augmentation is occasionally referred to as a regularisation
technique.
Regularisation decreases a model's variance by adding prior knowledge
(sometimes using shrinkage).
Increasing training data (using augmentation) also decreases a model's
variance.
Data augmentation is also a form of adding prior knowledge to a model.

### Citing

If you use mixup in a scientific publication, then consider citing the original paper:

mixup: Beyond Empirical Risk Minimization

By Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz

[https://arxiv.org/abs/1710.09412](https://arxiv.org/abs/1710.09412)

I have no affiliation with MIT, FAIR or any of the authors.


## Roadmap

 * Improve docs
   * Add before and after mixup plots for iris data
   * Add more detailed examples
     * Different data types e.g. image, temporal etc
     * Different parameters
 * Correct integer pandas data frame index assumption
   * Non-integer indices are currently unsupported
 * Add my time series mixup variant
   * Applies mixup technique to two time series separated by 'time_diff' period
   * Implemented and tested in this
     [encoder decoder Jupyter notebook](https://github.com/makeyourownmaker/CambridgeTemperatureNotebooks/blob/main/notebooks/encoder_decoder.ipynb)
 * Add label preserving option
 * Add support for mixing within the same class
   * Usually doesn't perform as well as mixing within all classes
   * May still have some utility e.g. unbalanced data sets


## Alternatives

Other implementations:
 * [pytorch from hongyi-zhang](https://github.com/hongyi-zhang/mixup)
 * [pytorch from facebookresearch](https://github.com/facebookresearch/mixup-cifar10)
 * [keras from yu4u](https://github.com/yu4u/mixup-generator)
 * [mxnet from unsky](https://github.com/unsky/mixup)
 * [An R package inspired by 'mixup: Beyond Empirical Risk Minimization'](https://github.com/makeyourownmaker/mixup)


## See Also

Discussion:
 * [inference.vc](https://www.inference.vc/mixup-data-dependent-data-augmentation/)
 * [Openreview](https://openreview.net/forum?id=r1Ddp1-Rb)
 
Closely related research:
 * [Manifold Mixup: Better Representations by Interpolating Hidden States](https://arxiv.org/abs/1806.05236)
 * [MixUp as Locally Linear Out-Of-Manifold Regularization](https://arxiv.org/abs/1809.02499)

Loosely related research:
 * [Label smoothing](https://arxiv.org/pdf/1701.06548.pdf)
 * [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)


## Contributing

Pull requests are welcome.  For major changes, please open an issue first to discuss what you would like to change.


## License
[GPL-3](https://www.gnu.ong/licenses/old-licenses/gpl-3.0.en.html)


<!-- Badges -->

[pypi-image]: https://img.shields.io/pypi/v/mixupy
[pypi-url]: https://pypi.org/project/mixupy/
[build-image]: https://github.com/makeyourownmaker/mixupy/actions/workflows/build.yml/badge.svg
[build-url]: https://github.com/makeyourownmaker/mixupy/actions/workflows/build.yml
[coverage-image]: https://codecov.io/gh/makeyourownmaker/mixupy/branch/main/graph/badge.svg
[coverage-url]: https://codecov.io/gh/makeyourownmaker/mixupy
[quality-image]: https://api.codeclimate.com/v1/badges/3130fa0ba3b7993fbf0a/maintainability
[quality-url]: https://codeclimate.com/github/makeyourownmaker/mixupy
