
The project focuses on finding a model capable of predicting the response to immunotherapy, given the T-cell repertoire sequencing (comprising the TRA repertoire and the TRB repertoire) from a patient, pre-treatment.

The preliminary phase of the project included sequencing the T-cell repertoire of GBM-induced mice, alignment, identifying unique peptides for each mouse, and calculating 31 biochemical properties of the unique peptides.

Subsequently, further processing of the sequencing data was performed, initial processing using the K-Means algorithm to obtain a dataset suitable for machine learning, and the dataset was entered into an SQL database.

In the next phase, we added a new feature that focused on a specific biochemical property- peptide hydrophobicity. several machine learning algorithms were tested:

Perceptron - a linear separation algorithm.

SVM - a linear separator that creates the largest possible margin between examples from different classes for training examples represented as vectors in linear space.

KNN - classifies an example according to the K nearest examples based on Euclidean distance.

Random Forest - classifies according to a large number of independent decision trees.

XgBoost - builds a random forest so that each tree improves itself using the previous trees.

In the final phase, the performance of the different algorithms was examined, addressing the following questions:

Is the data learnable?
What is the optimal learning algorithm?
Is there a difference between the TRA and TRB repertoires?
Did adding the hydrophobicity feature contribute to or hinder learning?

The results and detailed explanations are in the final report file. 
Data is available upon request.
