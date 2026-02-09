[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

# ICSTR-PINN-NMPC

Physics-Informed Neural Network (PINN) based Nonlinear Model Predictive Control (NMPC) for Isothermal Continuous Stirred Tank Reactor (CSTR)

## Overview

This project implements a Physics-Informed Neural Network (PINN) to model an isothermal CSTR and uses it within a Nonlinear Model Predictive Control (NMPC) framework for setpoint tracking.


## Physics Model

The CSTR follows the first-order reaction mass balance:dC_A/dt = u*(C_Ai - C_A) - k*C_A

Where:
- **C_A**: Concentration of species A
- **u**: Dilution rate (control input)
- **C_Ai**: Inlet concentration (1.0)
- **k**: Reaction rate constant (0.028)

Author
Emmanuel Alao
