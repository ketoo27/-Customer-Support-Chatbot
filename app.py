from flask import Flask, render_template, request
import re
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


# Initialize the Flask app
app = Flask(__name__)

# Load the saved model, vectorizer, and label encoder
log_reg_model = joblib.load('log_reg_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define responses for each intent
responses = {
    'cancel_order': "Your order has been canceled.",
    'complaint': "I'm sorry to hear that you have a complaint. Let me help you with that.",
    'contact_customer_service': "You can reach customer service via email or phone.",
    'track_order': "You can track your order by visiting the tracking page.",
    'payment_issue': "It looks like there was a payment issue. Please check your payment method.",
    'get_refund': "You can request a refund by filling out the form on our website.",
    # Add more responses for each intent as needed
}

# Function to clean text and get chatbot response
import re

# Clean the user input text
def clean_text(text):
    # Make sure text is a string
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        return text
    else:
        return ''


def get_response(user_input):
    # Clean the user input
    user_input_cleaned = clean_text(user_input)
    
    # Vectorize the input using the saved TF-IDF model
    user_input_vectorized = tfidf_vectorizer.transform([user_input_cleaned])
    
    # Predict the intent using the Logistic Regression model
    predicted_intent_index = log_reg_model.predict(user_input_vectorized)[0]
    
    # Convert the predicted index back to the intent label
    predicted_intent = label_encoder.inverse_transform([predicted_intent_index])[0]
    
    # Get the chatbot's response based on the predicted intent
    response = responses.get(predicted_intent, "Sorry, I didn't understand that.")
    
    return response

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle user input and return a chatbot response
@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_input = request.form['user_input']  # Get input from the form
    response = get_response(user_input)      # Get chatbot response
    return render_template('index.html', user_input=user_input, bot_response=response)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
