# NovaWebApp
Welcome to the official repository of tools and web applications developed by the Novalix Computational Chemistry Team

## Overview
In DEL screening, medicinal chemists often have to decide which library is the most appropriate in terms of both chemical diversity and target addressability (i.e. compatibility profile with the given target). 
To address this, we developed a tool that enables systematic quantification of both parameters, using BM-scaffold analysis and Machine Learning.

**Preprint 📝📈**:  https://doi.org/10.26434/chemrxiv-2024-9fm01-v2 

✨Update: We are happy to share that this work has been accepted for publication in *ACS Medicinal Chemistry Letters*

**Publication 📑✅**:  https://doi.org/10.1021/acsmedchemlett.4c00505  

<br />

## Link to Web Application 🖥️

https://novalix-novawebapp.hf.space/ 

## Key Features
- **NovaML:**
	- Build your own machine learning model with your data
	- Use your machine learning model to predict properties on your own data
- **NovaDel Analyzer:**
	- Use the TMAP algorithm to visualize your DEL in a chemical space
	- Use the Venn diagram algorithm to evaluate the structure innovativeness of your scaffolds
	- Evaluate the target adressability of your DEL with your own machine learning model

## How to install in your local PC
1. **Clone the Repository:**
	```
	git clone https://github.com/novalixofficial/NovaWebApp.git
 	cd NovaWebApp
	```
2. **Install Dependencies:**
	```
	conda env create -f environment.yml
 	conda activate streamlit_env
	```
3. **Run the NovaWeb App:**
	```
 	chmod u+x run.sh
	./run.sh
	```
 4. **Copy the URL in your Browser (Chrome or Edge):**
	```
 	http://localhost:1238
	```
⚠️ When running NovaDEL Analyzer, be aware that if you want to predict the addressability you will need the model generated in NovaML or your own model

## Licence
This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or suggestions, open an issue.
