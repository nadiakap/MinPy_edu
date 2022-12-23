# MinPy_edu - optimization library for research and education

Herein we provide examples of implementation of the theoretical approach described in "Nonlocal Optimization Methods Based on Potential Theory" by Kaplinskii et.al. (original paper can be found in references - kapl.pdf). Different aspects of this approach are discussed in the following papers which are all available in references: kapl_first_order.pdf, kapl_second_order.pdf
A.I. Kaplinskii and A.I. Propoi. Nonlocal optimization methods that use potential theory. Avtomat. i Telemekh., 7:55-65, 1993.
A.I. Kaplinskii and A.I. Propoi. Second-order refinement conditions in nonlocal optimization methods that use potential theory. Avtomat. i Telemekh., 8:104-117, 1994.
(references from stochastic programming bibliography: http://www.eco.rug.nl/mally/biblio/SPlist.html)

For application of the methods in machine learning see this paper:
DOI: https://doi.org/10.17308/sait.2018.4/1261

This library is under construction! We plan to add more building blocks over time.

Please check minpy_getting_started_example.py to see how to construct custom algorithms using minpy building blocks.
Currently the minpy library consists of two python files, minpy.py ( building block for the algorithm) and optfun.py (test functions).

Supporting paper from scipy 2022 conference: https://doi.org/10.25080/majora-212e5952-019
Supporting poster from Scipy 2022 conference can be found in references.

Prerequisites:

1. Python 3.6+
2. Scipy, Numpy, Pandas, Matplotlib, Jupiter notebook ( or Anaconda - all required packages are part of Anaconda)

Running instructions:

1. Clone minpy_edu repo
2. Install Jupyter notebook
3. Run minpy_getting_started.ipynb
