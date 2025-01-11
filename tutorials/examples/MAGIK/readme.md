# MAGIK

MAGIK is a geometric deep learning approach for the analysis of dynamical properties from time-lapse microscopy.
Here we provide the code as well as instructions to train models and to analyze experimental data.

# Getting started

## Installation from PyPi

MAGIK requires at least python 3.6.

To install MAGIK you must install the [Deeptrack](https://github.com/softmatterlab/DeepTrack-2.0) framework. Open a terminal or command prompt and run:

    pip install deeptrack

## Software requirements

### OS Requirements

MAGIK has been tested on the following systems:

- macOS: Monterey (12.2.1)
- Windows: 10 (64-bit)

### Python Dependencies

```
tensorflow
numpy
matplotlib
scipy
Sphinx==2.2.0
pydata-sphinx-theme
numpydoc
scikit-image
tensorflow-probability
pint
pandas

```

If you have a very recent version of python, you may need to install numpy _before_ DeepTrack. This is a known issue with scikit-image.

## It's a kind of MAGIK...

To see MAGIK in action, we provide an [example](//github.com/softmatterlab/DeepTrack-2.0/blob/develop/examples/MAGIK/) based on live-cell migration experiments. Data courtesy of Sergi Masó Orriols, [the QuBI lab](https://mon.uvic.cat/qubilab/).

## Cite us!

If you use MAGIK in your project, please cite our article:

```
Jesús Pineda, Benjamin Midtvedt, Harshith Bachimanchi, Sergio Noé, Daniel  Midtvedt, Giovanni Volpe, and  Carlo  Manzo
"Geometric deep learning reveals the spatiotemporal fingerprint of microscopic motion."
arXiv 2202.06355 (2022).
https://arxiv.org/pdf/2202.06355.pdf
```

## Funding

This work was supported by FEDER/Ministerio de Ciencia, Innovación y Universidades – Agencia Estatal de Investigación
through the “Ram ́on y Cajal” program 2015 (Grant No. RYC-2015-17896) and the “Programa Estatal de I+D+i Orientada a los Retos de la Sociedad” (Grant No. BFU2017-85693-R); the Generalitat de Catalunya (AGAUR Grant No. 2017SGR940); the ERC Starting Grant ComplexSwimmers (Grant No. 677511); and the ERC Starting Grant MAPEI (101001267); the Knut and Alice Wallenberg Foundation (Grant No. 2019.0079).

## License

This project is covered under the **MIT License**.
