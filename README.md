# NovaWebApp
Welcome to the official repository of tools and web applications developed by the Novalix Computational Chemistry Team

## Overview
As demand of DNA Encoded Libraries in drug discovery programs continues to grow, we developed in this study a tool to guide medicinal chemists.


## Key Features
- **NovaML:**
	- Build yout own machine learning model with your data
	- Use your machine learning model to predict properties on your own data
- **NovaDel Analyzer:**
	- Use the TMAP algorithm to visualize your DEL in a chemical space
	- Use the Venn diagram algorithm to evaluate the structure novelty of your scaffolds
	- Evaluate the target adressability of your DEL with your own machine learning model

## Getting Started
1. **Clone the Repository:**
	'''bash
	git clone https://github.com/novalixofficial/NovaWebApp.git
	'''
2. **Install Dependencies:**
	'''bash
	conda env create -f environment.yml
	'''
3. **Run the NovaWeb App:**
	'''bash
	./run.sh

⚠️ When running NovaDEL Analyzer, be aware that if you want to predict the addressability you will need the model generated in NovaML or your own model

## Licence
This project is licensed under the Apache License. See the [LICENCE](LICENSE) file for details.

## Contact
For questions or suggestions, open an issue or contact us at [pschambel@novalix.com](mailto:pschambel@novalix.com).
