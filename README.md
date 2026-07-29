Tools for relative (electrical) calibration of seismic instrumentation.

[TOC]

## Overview

Tools work with both Guralp and Nanometrics digitizers and generate formats required by IDC:

* `calibration_analyzer`: broadband calibration analyzer (see [calibration_analyzer_help.txt](docs/calibration_analyzer_help.txt))
* `sine_analyzer`: sine-wave calibration analyzer(see [sine_analyzer_help.txt](docs/sine_analyzer_help.txt))

## Installation

### Basic

```bash
python3.12 -m venv ~/virtualenvs/stn
source ~/virtualenvs/stn/bin/activate
pip install git+https://github.com/nackerley/calibration-analyzer
```

Thereafter, to use the tools, you need to activate the environment, and deactivate it when you're done:

```bash
source ~/virtualenvs/stn/bin/activate
...
deactivate
```

### Development

Coding standards are enforced by `develop` environment in [pyproject.toml](pyproject.toml).

Please feel free to fix bugs and add new features using a
[feature-branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow).

If you've expanded the functionality of the package, please add pytests.

To set up a development environment:

``` bash
git clone https://github.com/nackerley/calibration-analyzer
cd station-tools
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --editable .[develop]
```

## Contact

Email: <nicholas.ackerley@nrcan-rncan.gc.ca>
