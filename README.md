# Machine Learning Methods for Domestic Energy Prediction for Small-Neighbourhoods at National Scales in England and Wales

## Overview

This repository contains the code to accompany the paper of the same name. An implementation of neighborhood energy modeling using the NEBULA dataset to predict domestic energy consumption at postcode level across England and Wales.

The NEBULA dataset and associated papers can be found at https://github.com/graceebc9/NebulaDataset

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/gracecolverd/UKPostcodePrediction.git

# Navigate to the project directory
cd UKPostcodePrediction

# Create a virtual environment (recommended)
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
UKPostcodePrediction/
├── src/
│   ├── column_settings.py      # Column configurations and data preprocessing
│   ├── conf.py                 # Configuration settings
│   ├── feature_gen.py         # Feature generation and engineering
│   ├── problem_definitions.py  # Problem space definitions
│   └── sobol.py               # Sobol sequence implementation
├── run_automl.py              # AutoML pipeline execution
├── run_feature_imp.py         # Feature importance analysis
└── run_gsa.py                 # Global sensitivity analysis
```

## Usage

```python
# Run AutoML pipeline
python run_automl.py

# Analyze feature importance
python run_feature_imp.py

# Perform global sensitivity analysis
python run_gsa.py
```

## License

This work is licensed under a Creative Commons Attribution 4.0 International License:

Creative Commons Attribution 4.0 International (CC BY 4.0)

Copyright (c) 2025 Grace Colverd

This work is licensed under the Creative Commons Attribution 4.0 International License. 
To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or 
send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

You are free to:
- Share: copy and redistribute the material in any medium or format
- Adapt: remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, and 
  indicate if changes were made. You may do so in any reasonable manner, but not in 
  any way that suggests the licensor endorses you or your use.

No additional restrictions — You may not apply legal terms or technological measures 
that legally restrict others from doing anything the license permits.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{colverd2025mlmethods,
  author = {Colverd, Grace},
  title = {Machine Learning Methods for Domestic Energy Prediction for Small-Neighbourhoods at National Scales in England and Wales},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/gracecolverd/UKPostcodePrediction}
}
```

## Summary

This project implements machine learning methodologies to predict domestic energy consumption patterns across England and Wales at the postcode level. The implementation includes:

- Automated machine learning pipeline for model selection and optimization
- Feature importance analysis to identify key predictors of energy consumption
- Global sensitivity analysis using Sobol sequences
- Comprehensive data preprocessing and feature engineering pipeline
- Scalable methods for national-level predictions at neighborhood granularity

The codebase is structured to support reproducible research and can be extended for similar prediction tasks in other geographical contexts.

## Contact

Grace Colverd
- GitHub: @graceebc9
- BlueSky: [BlueSky handle not provided]