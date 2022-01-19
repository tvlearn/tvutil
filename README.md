# tvutil

Utilities for pre-/postprocessing and visualization developed and/or commonly used by the `tvlearn` community.


## Installation

We recommend [Anaconda](https://www.anaconda.com/) to manage the installation. To create a new environment that bundles all packages required, run:

```bash
$ conda create -n tvutil pip
```

To install the packages required, run

```bash
$ pip install -r requirements.txt
```

Finally, to install `tvutil` run:

```bash
$ python setup.py install develop
```


__Remark__
*  If you are not planning to contribute and only want to use the implementations provided in this repo, you could comment out the packages `black`, `pylama`, `mypy`, `pytest` in the `environment.txt` file.
*  Parts of the implementation do not require a PyTorch installation, i.e. you may optionally also comment `torch` out in `environment.txt`.
