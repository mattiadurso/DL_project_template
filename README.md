# DL Project Template
This is my repo for DL projects. It contains the structure and most of the base code used to build up a DL project. I hope to mantain this during time.

Reference for coding -> https://goodresearch.dev/_static/book.pdf <br>


Repo structure: 
<pre>
.
├── config              # .yaml files used to configure experiments.
├── data                # Locally, data is here. Data is not uploaded.
├── docs                # All docs needed to run and undestand the project.
├── models              # Trained models weights.           
├── scripts             # Scripts and notebooks.
    ├── data                # Scripts to pre-process and load data.
    ├── models              # Models' classes.
    ├── constants.py        # File in which save constants, like WandB key. Create yours.
├── src                 # Reusable Python modules, need to install.
    └── utils.py            # Script with all the ancillar functions (e.g. train() or select_optim()).
├── .gitignore 
├── environment.yml    # All python libraries needed (curently is set randomly).
├── README.md          # The readme you are looking at.
└── setup.py           # Make this project pip installable with `pip install -e`
</pre>

Install src functions with: <br>
```
 pip install -e .
```
Some useful tools:
<pre>
- Running ```pylint``` for formatting. 
- Use Google style to document code and Sphinx to generate docs.
- ```vulture``` a Python package to identify dead code in your codebase

</pre>