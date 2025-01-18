# disease-detection-using-ML
save the file with same name as give : symptom_based_DD[1].ipynb




import tensorflow as tf  # Import TensorFlow for building the neural network
from tensorflow.keras import datasets, layers, models  # Import necessary modules for building the model
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualizing the results
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For image preprocessing
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
import os
import warnings
warnings.filterwarnings("ignore")
dataset = pd.read_csv('/Users/pavan/Desktop/major Proj/symptom_text/Training.csv')
dataset.head() #shows number of columns and rows
dataset.shape  ##shows number of columns and rows
len(dataset[ 'prognosis'] .unique())  #show size of unique names in particular column
dataset['prognosis'].unique()    #show unique names in particular column
#TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X = dataset.drop("prognosis", axis=1)  # zero (0) for row wise 1 for column wise
y = dataset ['prognosis']
X
y
le = LabelEncoder()   #label encoder converts strings into numerical values 
le. fit(y)
Y = le. transform(y)
Y
X_train,X_test,y_train,y_test = train_test_split(X,Y ,test_size=0.3, random_state=20) 
#split the data into four 4 variables xtrain,x test,ytrain,ytest testsize is to detremine the percentage to test and train 0.3 represents 70%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y ,test_size=0.3, random_state=20) 
#split the data into four 4 variables xtrain,x test,ytrain,ytest testsize is to detremine the percentage to test and train 0.3 represents 70%
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix 
import numpy as np

#creating a dictionary to store models

models = {
"SVC": SVC(kernel= 'linear'),
"RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
'GrandientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
"KNeighbors": KNeighborsClassifier(n_neighbors=5),
"MultinomialNB": MultinomialNB(),
"DecisionTree": DecisionTreeClassifier(random_state=42)
}
for model_name, model in models.items():
    print (model_name, ":", model)
for model_name, model in models.items():
    # train model
    model. fit(X_train,y_train)
    # test model
    predictions = model.predict(X_test)
    # calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    # calculate confusion matrix
    cm = confusion_matrix (y_test,predictions)
    print(f" {model_name} accuracy : {accuracy}")
    print (f" {model_name} Confusion Matrix:")
    print(np.array2string(cm, separator=', '))
# Initialize Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Fit the model on training data
decision_tree.fit(X_train, y_train)

# Predict on test data
y_pred = decision_tree.predict(X_test)

# Calculate and print accuracy score in percentage
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Decision Tree Classifier: {accuracy * 100:.2f}%")

# saving model
import pickle
pickle. dump (decision_tree, open ("/Users/pavan/Desktop/major Proj.pkl", 'wb'))
# Load model
decision_tree_model = pickle. load (open ("/Users/pavan/Desktop/major Proj.pkl", 'rb'))
decision_tree_model.predict(X_test.iloc[0]. values.reshape(1,-1))
y_test[0]

# test 1
print("Predicted Label :", decision_tree_model.predict(X_test.iloc[0].values.reshape(1,-1)))
print("Actual Label :", Y_test[0])
# test 2
print("Predicted Label :", decision_tree_model.predict(X_test.iloc[12].values.reshape(1,-1)))
print("Actual Label :", Y_test[12])
sym_des = pd.read_csv('/Users/pavan/Desktop/major Proj/symptom_text/symtoms_df.csv')
precautions = pd.read_csv("/Users/pavan/Desktop/major Proj/symptom_text/precautions_df.csv")
workout = pd.read_csv("/Users/pavan/Desktop/major Proj/symptom_text//workout_df.csv")
description = pd. read_csv("/Users/pavan/Desktop/major Proj/symptom_text//description.csv")
medications = pd. read_csv("/Users/pavan/Desktop/major Proj/symptom_text//medications.csv")
diets = pd. read_csv("/Users/pavan/Desktop/major Proj/symptom_text//diets.csv")
np.zeros(10)
#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == predicted_disease]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout



symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[decision_tree_model.predict([input_vector])[0]]
# Test 1
# Split the user's input into a list of symptoms (assuming they are comma-separated) # itching,skin_rash,nodal_skin_eruptions
symptoms = input("Enter your symptoms.......")
user_symptoms = [s.strip() for s in symptoms.split(',')]
# Remove any extra characters, if any
user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
predicted_disease = get_predicted_value(user_symptoms)

desc, pre, med, die, wrkout = helper(predicted_disease)

print("=================predicted disease============")
print(predicted_disease)
print("=================description==================")
print(desc)
print("=================precautions==================")
i = 1
for p_i in pre[0]:
    print(i, ": ", p_i)
    i += 1

print("=================medications==================")
for m_i in med:
    print(i, ": ", m_i)
    i += 1

print("=================workout==================")
for w_i in wrkout:
    print(i, ": ", w_i)
    i += 1

print("=================diets==================")
for d_i in die:
    print(i, ": ", d_i)
    i += 1
# Step 1: Load and Preprocess the Dataset
# Assuming you have a dataset of images organized in subfolders for each class (disease).
# Use ImageDataGenerator to preprocess images and apply augmentation for better training results.
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescaling pixel values to the range [0, 1]
    shear_range=0.2,  # Random shear transformation for augmentation
    zoom_range=0.2,  # Random zoom transformation for augmentation
    horizontal_flip=True  # Random horizontal flipping for augmentation
)
test_datagen = ImageDataGenerator(rescale=1./255)  # Rescale test images


# Load the data using the flow_from_directory method
train_generator = train_datagen.flow_from_directory(
    '/Users/pavan/Desktop/major Proj/skin-disease-datasaet/train_set',  # Path to training dataset directory
    target_size=(64, 64),  # Resize images to 64x64 pixels
    batch_size=32,  # Number of images to process in a batch
    class_mode='categorical'  # For multi-class classification
)
validation_generator = test_datagen.flow_from_directory(
    '/Users/pavan/Desktop/major Proj/skin-disease-datasaet/test_set',  # Path to validation dataset directory
    target_size=(64, 64),  # Resize images to 64x64 pixels
    batch_size=32,  # Number of images to process in a batch
    class_mode='categorical'  # For multi-class classification
)
from tensorflow.keras import models
from tensorflow.keras import layers

# Step 2: Define the CNN Model
model = models.Sequential()
# Add a convolutional layer with 32 filters, kernel size of (3, 3), and 'relu' activation function
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))  # Max pooling layer with pool size of (2, 2)

# Add another convolutional layer with 64 filters
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Add another convolutional layer with 128 filters
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Flatten the output from the convolutional layers to feed into a fully connected layer
model.add(layers.Flatten())
# Add a dense fully connected layer with 512 units and 'relu' activation
model.add(layers.Dense(512, activation='relu'))
# Output layer with softmax activation function for multi-class classification
# Assuming there are 5 different diseases to classify
model.add(layers.Dense(8, activation='softmax'))
# Step 3: Compile the Model
# Use Adam optimizer, categorical crossentropy loss function, and accuracy metric for multi-class classification
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

import tensorflow as tf

# Enable eager execution (necessary for NumPy conversion)
tf.compat.v1.enable_eager_execution()

# # Step 4: Train the Model
# # Fit the model on the training data and validate it using the validation data
# history = model.fit(
#     train_generator,
#     steps_per_epoch=50,  # Number of steps per epoch (number of batches)
#     epochs=10,  # Number of times to iterate over the dataset
#     validation_data=validation_generator,
#     validation_steps=50  # Number of validation steps
# )
# Step 4: Train the Model
# Fit the model on the training data and validate it using the validation data

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Use the length of the generator instead
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)  # Similarly for validation
)

# Step 5: Save the Model
# Save the trained model to a file so it can be used later for predictions
model.save('disease_detection_model.h5')  # Save the model to an H5 file
# Step 6: Print Accuracy
# Print the final training and validation accuracy
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Evaluate the model on the validation dataset
loss, accuracy = model.evaluate(validation_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
from tensorflow.keras.models import load_model

# Load the model
loaded_model = load_model('disease_detection_model.h5')

# Step 6: Test the Model on a New Image
# Load a new image for testing (make sure the image is preprocessed similarly as the training images)
from tensorflow.keras.preprocessing import image
img = image.load_img('/Users/pavan/Desktop/major Proj/skin-disease-datasaet/test_set/FU-athlete-foot/FU-athlete-foot (1).jpg', target_size=(64, 64))
img_array = image.img_to_array(img)  # Convert the image to an array
img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size
img_array = img_array / 255.0  # Rescale the image

# Predict the disease class
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)  # Get the index of the highest probability
# Step 2: Get the number of classes
num_classes = model.output_shape[-1]  # The last dimension of the output layer is the number of classes

# Step 3: Print the number of classes
print(f"The model has {num_classes} output classes.")
# Get the class names from the train_generator
class_names = list(train_generator.class_indices.keys())

# Print the class names
print("Class names:", class_names)

# Step 1: Load the Trained Model
# Assuming the model is saved as 'disease_detection_model.h5'
model = tf.keras.models.load_model('disease_detection_model.h5')

# Step 2: Define class names
class_names = ['BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles']  # Update based on your dataset

# Step 3: Function to preprocess and predict disease from user input image
def predict_disease(img_path):
    # Load the image with the target size (64x64) as used in training
    img = image.load_img(img_path, target_size=(64, 64))
    
    # Convert the image to an array
    img_array = image.img_to_array(img)
    
    # Add an extra dimension for batch size (the model expects a batch of images)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Rescale the image (similar to what was done during training)
    img_array = img_array / 255.0
    
    # Predict the disease class (output from the model)
    prediction = model.predict(img_array)
    
    # Get the index of the predicted class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    
    # Map the predicted class index to the corresponding disease name
    disease_name = class_names[predicted_class[0]]
    
    return disease_name

# Step 4: User Input and Prediction
uploaded_image = input("Enter the path of the image to classify: ").strip()  # User input for image path

if os.path.exists(uploaded_image):
    predicted_disease = predict_disease(uploaded_image)
    print(f"The uploaded image is classified as: {predicted_disease}")
else:
    print("Invalid image path. Please try again.")
import sklearn
print(sklearn.__version__)
import tensorflow
print(tensorflow.__version__)

