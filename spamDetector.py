import streamlit as st
import pickle

# Load the saved Naive Bayes model and CountVectorizer object
model = pickle.load(open('spam123.pkl','rb'))
cv=pickle.load(open('vec123.pkl','rb'))

# Define the main function for the Streamlit app
def main():
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify email as spam or ham")
    st.subheader("Classification")
    user_input=st.text_area("Enter an email to classify", height=150)
    if st.button("classify"):  # Create a button that triggers the classification process when clicked
        if user_input:      # Check if the user has entered any input
            data=[user_input]       # Store the user input in a list to prepare it for vectorization
            print(data)
            vec=cv.transform(data).toarray()    # Transform the input data using the loaded vectorizer (cv) and convert it to an array
            result=model.predict(vec)       # Use the loaded model to predict whether the email is spam or not
            if result[0]==0:        # Check the prediction result
                st.success("This is Not A Spam Email")  
            else:
                st.error("This is A Spam Email")
        else:
            st.write("Please enter an email to classify")
main()