File name: RETAIN_Data_Generator.ipynb
This is a detailed overview of the Python script designed to process the MIMIC-III dataset for use with the RETAIN-Keras framework. The script's purpose is to prepare longitudinal patient records for training machine learning models, specifically tailored for healthcare applications.
Script Overview
The script is a critical component in preparing healthcare data for predictive modeling. It facilitates the conversion of raw MIMIC-III dataset files into structured data suitable for training RETAIN-Keras models. This section provides an overview of the script's primary functions and operation.
Purpose
•	The primary goal of the script is to create longitudinal patient records.
•	It identifies patients with two or more encounters in the MIMIC-III dataset and organizes their data.
•	The output of the script consists of pickled pandas dataframes, dictionary mappings, and target labels.
The following parameters need to be changed accordingly:
•	ADMISSIONS.csv: The ADMISSIONS.csv file from MIMIC-III.
•	DIAGNOSES_ICD.csv: The DIAGNOSES_ICD.csv file from MIMIC-III.
•	PATIENTS.csv: The PATIENTS.csv file from MIMIC-III.
•	output directory: The directory where the script will save output files.
•	train data proportion: Proportion of data to be used for training (e.g., 0.7 for 70% training data).
Dataframes
•	data_train.pkl and data_test.pkl: Pickled dataframes for training and testing, containing patient codes and time-to-event sequences.
•	data_train_3digit.pkl and data_test_3digit.pkl: Pickled dataframes for training and testing, with 3-digit diagnosis codes.
•	target_train.pkl and target_test.pkl: Pickled dataframes containing target labels for training and testing.
Key Functions
•	convert_to_icd9(dx_str)
o	Maps an ICD diagnosis code to ICD9 format.
o	Handles both regular and "E" diagnosis codes.
•	convert_to_3digit_icd9(dx_str)
o	Rolls up a diagnosis code to 3 digits.
o	Accommodates both regular and "E" diagnosis codes.

Main Script
The main script operates as follows:
1.	Collecting Mortality Information
a.	Builds a map of patient IDs to their mortality status (1 for deceased, 0 for alive).
2.	Building pid-admission Mapping
a.	Maps patient IDs to admission IDs.
b.	Captures admission dates.
3.	Creating Admission dx Code Mapping
a.	Organizes diagnosis codes associated with each admission.
b.	Separates codes into full diagnosis codes and 3-digit codes.
4.	Building Ordered Visit Mapping
a.	Arranges patient visits chronologically.
b.	Creates mappings for both full codes and 3-digit codes.
5.	Creating Sequences
a.	Generates sequences of patient IDs, admission dates, diagnosis code sequences, and mortality labels.
6.	Mapping Code Strings to Integers
a.	Converts code strings to unique integer representations.
b.	Maintains a dictionary for code-to-integer mapping.
7.	To-Event Column Calculation
a.	Computes the "to_event" column, representing time to a reference date.
b.	Calculates "Time of visit" as a numeric column for admission time of day.
8.	Sorting Data
a.	Sorts dataframes based on the number of visits per patient.
9.	Train-Test Split
a.	Creates train and test data splits for model training and evaluation.
10.	Reverse Dictionaries
a.	Constructs dictionaries for mapping integer codes back to their original strings.
11.	Saving Data
a.	Outputs dataframes, dictionaries, and target labels to the specified output directory.

File name: RETAIN_train.ipynb
Script Overview
The Python script presented here is a crucial component for applying the RETAIN model to healthcare datasets. It facilitates data preprocessing, model construction, and training. Below are key aspects of the script's operation.
Purpose
The primary goal of the script is to implement the RETAIN model for predictive healthcare analytics. It combines medical code sequences, visit information, and optionally, numeric data and timestamps, to predict patient outcomes.
Parameters
1.	num_codes: Number of medical codes.
2.	numeric_size: Size of numeric inputs (0 if none).
3.	use_time: Flag to indicate the use of time input.
4.	emb_size: Size of the embedding layer.
5.	epochs: Number of training epochs.
6.	n_steps: Maximum number of visits after which data is truncated.
7.	recurrent_size: Size of the recurrent layers.
8.	path_data_train: Path to the training data.
9.	path_data_test: Path to the testing data.
10.	path_target_train: Path to the training target.
11.	path_target_test: Path to the testing target.
12.	batch_size: Batch size for training.
13.	dropout_input: Dropout rate for embedding.
14.	dropout_context: Dropout rate for context vector.
15.	l2: L2 regularization value.
16.	directory: Directory to save the model and log files.
17.	allow_negative: Flag to allow negative weights for embeddings/attentions.
Model Architecture
The RETAIN model architecture is a key component of the script. It involves the following key layers and operations:
1.	Embedding Layer: Converts medical codes into dense embeddings.
2.	Bidirectional LSTM Layers: Captures temporal dependencies in EHR data.
3.	Attention Mechanisms: Computes alpha (visit attention) and beta (code attention) weights.
4.	Context Vector Calculation: Combines alpha, beta, and embeddings to produce context vectors.
5.	Output Layer: Predicts patient outcomes using the context vectors.

Training and Callbacks
The script includes training functionality, where the model is compiled and trained on the provided data. Training involves the use of callbacks, including custom logging for metrics like ROC-AUC and PR-AUC.
Freeze padding
FreezePadding class is a custom constraint used in Keras for weight constraints in neural network layers. Specifically, it's used as a constraint for weight matrices. This constraint is applied to the last weight in the weight matrix of certain layers to ensure that this weight is "frozen" or set to specific values, typically near 0.
The purpose of the FreezePadding constraint is related to the RETAIN model architecture, which is used for predictive healthcare analytics. In this context, freezing the last weight to be near 0 serves a specific purpose, which may be explained by the following considerations:
a.	Embeddings: In neural network models, embeddings are often used to represent categorical data, such as medical codes. These embeddings are learned during training. By using this constraint, the model ensures that the last weight in the embeddings layer remains near 0, possibly to prevent negative embeddings.
b.	Preventing Negative Weights: In some applications, especially in healthcare analytics, negative weights in embeddings or attention mechanisms might not make intuitive sense. By "freezing" the last weight to be near 0, the model effectively prevents negative weights from affecting the representation of medical codes or attention scores.
c.	Control over Embeddings: Controlling the embeddings' properties can be important. By setting the last weight to near 0, you have more control over the embeddings' behavior, especially if you want to enforce specific constraints on the embeddings.
The reason for having two different weight constraint classes, FreezePadding and FreezePadding_Non_Negative, in the code depends on the specific requirements and design choices for the model. These constraints serve different purposes related to weight management in the RETAIN model:
a.	FreezePadding: This constraint is used to freeze the last weight in a weight matrix to be near 0. It allows the weight to be either positive or negative, effectively preventing large negative values but still allowing some flexibility in the weights. The FreezePadding constraint is applied when the allow_negative flag is set to True.
b.	FreezePadding_Non_Negative: This constraint is used when you want to enforce that the last weight in the weight matrix must be non-negative (greater than or equal to 0). It ensures that no negative values are present in the last weight. This constraint is applied when the allow_negative flag is set to False.

`get_importances` Function:
This function plays a crucial role in interpreting and extracting feature importances from a machine learning model. Its purpose is to shed light on which features (in this case, medical codes, or other relevant data) contributed the most to a model's prediction for a given patient. Here's a step-by-step explanation:
1. Input Parameters:
1.	`alphas`: This represents the attention scores or weights for each feature in the model's output. It reflects the importance of each feature at each time step during a patient's visits.
2.	`betas`: These are coefficients that weigh the contribution of embeddings (representations) of features.
3.	`patient_data`: The patient's data, which includes codes, numerics, and time information for each visit.
4.	`model_parameters`: Information about the model's architecture and settings.
5.	`dictionary`: A dictionary that maps feature indices to their names.
2. Initialization:
1.	The function initializes an empty list called `importances`, which will store dataframes for each visit's feature importances.
3. Loop Over Visits:
1.	The function iterates over each visit made by the patient.
4. Extract Relevant Data:
   For each visit, it extracts the following data:
1.	`visit_codes`: The medical codes associated with the visit.
2.	`visit_beta`: The beta coefficients specific to this visit.
3.	`visit_alpha`: The attention scores for this visit.
4.	`relevant_indices`: The indices of relevant features for this visit, including both codes and potentially numeric features.
5.	`values`: Initially set to "Diagnosed" for each code, and it may include numeric values if applicable.
6.	`values_mask`: A mask that converts "Diagnosed" values to 1.0 and leaves numeric values unchanged.
5. Feature Importance Calculation:
1.	The function calculates feature importance for this visit. It involves the following steps:
2.	Scaling beta coefficients by embedding weights.
3.	Scaling by alpha scores.
4.	Combining these to compute feature importances.

6. Dataframe Creation:
•	A dataframe is created for this visit's feature importances, including columns for feature status ("Diagnosed" or numeric), feature name, feature importance, visit importance, and time to the event.
7. Filtering and Sorting:
•	The dataframe is filtered to exclude "PADDING" features and is then sorted based on feature importance in descending order.
8. Appending to Importances List:
•	The resulting dataframe for this visit is added to the `importances` list.
9. Return Value:
•	After processing all visits, the function returns the list of dataframes (`importances`). Each dataframe provides insights into the importance of features for a specific visit, helping users understand why the model made certain predictions for that patient.

`get_predictions` Function:
This function is responsible for using a loaded machine learning model to make predictions on a given dataset. It serves as a bridge between the model and the data, facilitating the prediction process. Here's a detailed explanation:
1. Input Parameters:
•	`model`: The machine learning model that was loaded earlier.
•	`data`: The dataset on which predictions need to be made.
•	`model_parameters`: Information about the model's architecture and settings.
2. Data Generator:
•	The function initializes a data generator named `test_generator` using the `SequenceBuilder` class. This generator prepares batches of data for the model's prediction.
3. Predictions:
•	It uses the `model` to make predictions on the dataset using the `predict_generator` method. This method takes advantage of parallel processing and uses multiple workers to speed up predictions.
4. Return Value:
•	The function returns the predictions made by the model on the input data.

In summary, `get_importances` is primarily focused on interpreting and visualizing feature importances for a given patient's data, which can provide insights into why the model arrived at a particular prediction. On the other hand, `get_predictions` is more concerned with using the model to generate predictions for a dataset and returning those predictions for further analysis. Together, these functions contribute to the interpretability and analysis of the machine learning model's behavior on medical data.

Output
The provided output is the result of running the code on a specific patient. Let's break down the output step by step:
1.	Input Patient Order Number: The program asks the user to input a patient order number. In this case, the user entered 1, indicating they want to examine patient number 1.
2.	Patients Probability: The program calculates and displays the probability associated with the patient. In this case, the patient has a probability of approximately 0.5582. This probability likely represents the likelihood of a specific medical event or outcome occurring for this patient based on the model's predictions.
3.	Output Predictions? The program asks whether the user wants to see predictions for this patient's data. The user responded with "y," indicating they want to proceed.
Feature Importances for Each Visit:
The following table(s) provide feature importance for each visit made by the patient. Each table corresponds to a visit.
Columns Explanation:
1.	status: Indicates the status of the feature, which is "Diagnosed" in this case. This imply that the feature was diagnosed or observed during the patient's visit.
2.	feature: Represents the specific medical feature (e.g., medical code) that is being analyzed for importance.
3.	importance_feature: Shows the importance of the feature during this visit. This importance is calculated based on the model's attention mechanism and coefficients.
4.	importance_visit: Reflects the overall importance of this visit. It is calculated based on the model's attention mechanism.
5.	to_event: Indicates the time to the event, which is 0 in the first table and 1 in the second. This represents the time elapsed between visits.
Table 1 (Visit 1):
•	The table shows feature importances for the patient's first visit.
•	Some features have negative importances, while others have positive importances.
•	The importance_feature column provides the feature-specific importance.
•	The importance_visit column shows the overall visit importance.
Table 2 (Visit 2):
•	This table provides feature importances for the patient's second visit.
•	Like the first visit, it displays feature importances for various medical features.
•	The values in the importance_feature column indicate the importance of each feature for this visit.
•	The importance_visit column reflects the overall importance of this visit.
These tables help interpret why the model made certain predictions for this patient. Features with higher positive or negative importances are considered more influential in making predictions. The "to_event" column might be relevant for time-sensitive predictions, indicating when the event occurred or will occur.
In summary, this output provides insights into how the model arrived at its predictions for a specific patient by examining the importance of different medical features during each visit. It allows healthcare professionals to better understand the factors influencing the model's predictions and make informed decisions based on the model's insights.
