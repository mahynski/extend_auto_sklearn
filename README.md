extend_auto_sklearn
===================

Automatically extend the auto-sklearn package with custom classes.
----

1. Create new conda environment to work in.
```bash
$ conda create -n automl python=3.7
$ conda activate automl
```
2. Install [auto-sklearn](https://automl.github.io/auto-sklearn/master/).
```bash
$ curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
$ pip install auto-sklearn
```
3. Clone this repository.
4. From notebook or script, simply import the library to automatically extend auto-sklearn.
```python
import extend_auto_sklearn
```
