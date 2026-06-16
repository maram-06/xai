## XAI Library
the library version of XAIExplainer class originally developed for An Explainable AI Dashboard for Traffic and Air Quality Analysis in Istanbul project

the purpose of this library is to provide a tool to convert machine learning model results into narratives to explain which features affected the result of the model. it uses the generic SHAP explainer, so the explainer changes according to the chosen model (e.g. ExactExplainer)

### Requirements Installation
run `pip install -r requirements.txt` on the terminal

### How To Use
include the following line in your code
`from xai import XAIExplainer`
if the library folder is under the main project directiory, you may need to include `-m` when running from the terminal

initialize the explainer once by instanciating the class then run the explainer function. to store into a postgreSQL database, you can call the function save_to_db.

### Testing
to run the test available in the library at tests/test.py, on the terminal run
`python -m tests.test`
