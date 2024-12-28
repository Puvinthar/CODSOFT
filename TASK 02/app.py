import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
iris_data = pd.read_csv('IRIS.csv',engine='python')

# Preprocess the data
label_encoder = LabelEncoder()
iris_data['species'] = label_encoder.fit_transform(iris_data['species'])

# Standardize the features
scaler = StandardScaler()
iris_data_scaled = scaler.fit_transform(iris_data.drop(columns='species'))

# Split the data
X = iris_data.drop(columns='species')
y = iris_data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Streamlit app
st.title("Iris Flower Classification")

st.sidebar.header("User Input Features")
st.sidebar.markdown("""[Example CSV input file](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combine user input features with entire iris dataset
# This will be useful for the encoding and scaling steps
iris = pd.concat([input_df, iris_data.drop(columns='species')], axis=0)

# Encode and scale the input
iris = pd.DataFrame(scaler.fit_transform(iris), columns=iris.columns)
input_scaled = iris[:1]

# Predict the classification
prediction = clf.predict(input_scaled)
prediction_proba = clf.predict_proba(input_scaled)

st.subheader('User Input features')
if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

st.subheader('Prediction')
iris_species = label_encoder.inverse_transform(prediction)
st.write(iris_species)

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Visualization
if st.checkbox('Show pairplot of the dataset'):
    sns.pairplot(iris_data, hue='species', palette='Set1')
    st.pyplot(plt.gcf())

if st.checkbox('Show heatmap of the dataset'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(iris_data.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt.gcf())
