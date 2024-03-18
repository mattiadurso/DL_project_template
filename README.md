# DL Project Template
This is my repo for DL projects. It contains the structure and most of the base code used to build up a DL project. I hope to mantain this during time. LAST UPDATE 18/03/2024.


Structure: 
<pre>
.
├── config              # .yaml files used to configure experiments
├── data                # locally, data is here. Data is not uploaded
├── docs                # all docs needed to run and undestand the project
├── models              # models weights           
├── notebooks           # notebooks
├── src                 # all the python scripts
    ├── data            # scripts to pre-process and load data
    ├── models          # models' classes
    ├── constants.py    # file in which save constants, like WandB key. Create yours.
    └── utilis.py       # script with all the ancillar functions (e.g. train() or select_oprim())
├── LICENSE
├── README.md
└── requirements.txt    # all python libraries needed (curently is set randomly)
</pre>
