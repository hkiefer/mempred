# mempred

 <img src="./example/logo_mempred.png"> 
<!--![Logo](./example/logo_mempred.png) -->


Python 3 tool to analyze and predict arbitrary one-dimensional time-series data. In the mempred.py model, extraction tools for the memory kernel including free energy-determination are included. The extraction is proceeded either by the Volterra method [memtools](https://github.com/jandaldrop/memtools) (© Jan Daldrop, Florian Brüning) or the discrete estimation method. 

For more information about the theory and algorithms, Please read and cite the following publication,

 (publication in preparation).

Required librarys: numpy, pandas, scipy, matplotlib,siml,sympy,tidynamics, (yfinance, wwo_hist for data loading)

## Setup

To run this code, install mempred dependencies first

```sh
$ pip3 install .
```

Make sure, Python3 is installed on your local machine.

See the example jupyter notebooks for instructions.
See `example/Example Memory Extraction.ipynb` for an introduction of the Volterra and Discrete Estimation extraction schemes, and `example/Example Stock Price.ipynb` and `example/Example Temperature.ipynb` for an introduction of the extraction and prediction for exemplary finance and weather data. 



