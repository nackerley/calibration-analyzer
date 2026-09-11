# Calibration Analyzer

Tools for relative (electrical) calibration of seismic instrumentation.

> [!IMPORTANT]
> The primary development of this project takes place on a private GitLab server that is automatically push-mirrored to GitHub.

An investigation of short-period seismometer temperature susceptibility making use of calibration analyzer for arbitrary input signals was initially presented at CTBTO SnT 2023: <https://conferences.ctbto.org/event/23/contributions/4925/>. A paper on the same subject has been accepted for publication in a [Seismological Research Letters (SRL) Focus Section on Measuring and Monitoring Seismic Instrumentation](https://www.seismosoc.org/publications/calls-papers/srl-call-for-papers-14/).

* [Overview](#overview)
* [Installation](#installation)
  * [Basic](#basic)
  * [Development](#development)
* [Contributing](#contributing)
* [Citation](#citation)
* [License](#license)
* [Contact](#contact)

## Overview

These tools work with both Guralp and Nanometrics digitizers, and generate formats required by IDC:

* `arbitrary_analyzer`: calibration analyzer for arbitrary input signals (see [arbitrary_analyzer_help.txt](docs/arbitrary_analyzer_help.txt))
* `sine_analyzer`: calibration analyzer for sinusoidal input signals (see [sine_analyzer_help.txt](docs/sine_analyzer_help.txt))

Examples can be found at [examples](examples).

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
cd calibration-analyzer
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --editable .[develop]
```

## Contributing

* **Issues**: Please report bugs or request features via GitHub Issues.
* **Pull Requests**: Do not open pull requests on GitHub. Because this is a push mirror, any changes made to branches will be automatically overwritten by GitLab during the next synchronization.
* **Code**: If you wish to contribute code, please open an issue to discuss your proposal.

## Citation

Please cite as:

> Ackerley, N. and Gias, Z. (2026). Investigating temperature dependence of frequency response of short-period seismometers using pole-zero fitting. Seism. Res. Lett., in review.

## License

[GNU General Public License, Version 3, 29 June 2007](LICENSE.md).

Copyright (c) His Μajesty the Κing in Right of Canada, as represented by the
Minister of Natural Resources, 2026.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

## Contact

Nick Ackerley: <nicholas.ackerley@nrcan-rncan.gc.ca>
