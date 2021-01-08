extend_auto_sklearn
===================

Automatically extend the [auto-sklearn](https://automl.github.io/auto-sklearn/master/) package with custom classes.

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

1. Create new conda environment to work in.
~~~ bash
$ sudo apt-get install build-essential swig # ONLY IF RUNNING UBUNTU
$ conda create -n automl python=3.7
$ conda activate automl
$ conda install gxx_linux-64 gcc_linux-64 swig
~~~

2. Install [auto-sklearn](https://automl.github.io/auto-sklearn/master/).
~~~ bash
$ curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
$ pip3 install auto-sklearn
~~~

3. Clone this repository.
~~~ bash
$ git clone https://github.com/mahynski/extend_auto_sklearn
~~~

4. Simply add this directory to your PYTHONPATH, or locally in each instance (i.e., sys.path.append()) and import the model, as usual, to automatically extend auto-sklearn.
~~~ bash
$ echo 'export PYTHONPATH=$PYTHONPATH:/path/to/module/' >> ~/.bashrc
$ source ~/.bashrc
~~~

~~~ python
import extend_auto_sklearn
~~~
