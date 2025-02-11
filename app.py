import streamlit as st
import re
import joblib

# Load the saved model, vectorizer, and label encoder
log_reg_model = joblib.load('model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define chatbot responses for each intent
responses = {
    'cancel_order': "Your order has been canceled. Would you like confirmation of this cancellation?",
    'change_order': "To modify your order, please provide your order number and the changes you'd like to make.",
    'change_shipping_address': "You can update your shipping address in your account settings or by contacting our support team.",
    'check_cancellation_fee': "Cancellation fees depend on your order status. You can view details in our cancellation policy.",
    'check_invoice': "Your invoice is available in your account's order history section. Should I email you a copy?",
    'check_payment_methods': "We accept Visa, Mastercard, PayPal, and Apple Pay. You can manage payment methods in your account.",
    'check_refund_policy': "Our standard refund policy allows returns within 30 days of delivery. See full details in our help center.",
    'complaint': "I'm sorry to hear about this issue. Let me escalate this to our resolution team immediately.",
    'contact_customer_service': "You can reach our support team 24/7 at support@company.com or +1-800-123-4567.",
    'contact_human_agent': "Connecting you with a live agent... Please hold for the next available representative.",
    'create_account': "You can create an account using your email or social media profiles. Would you like me to guide you?",
    'delete_account': "Account deletion can be processed through your profile settings. Confirm deletion via email verification.",
    'delivery_options': "We offer standard (3-5 days), express (1-2 days), and same-day delivery (select areas).",
    'delivery_period': "Current delivery estimates are 3-5 business days. Tracking updates will be sent to your email.",
    'edit_account': "You can update your personal information and preferences in the 'My Profile' section.",
    'get_invoice': "Your invoice has been sent to your registered email. Check spam folder if not received.",
    'get_refund': "Refund requests can be submitted through your order details page. Processing takes 5-7 business days.",
    'newsletter_subscription': "Manage email preferences in account settings. Unsubscribe links are in all marketing emails.",
    'payment_issue': "Payment failures can occur due to expired cards or bank declines. Please verify your payment details.",
    'place_order': "Your cart is ready for checkout. Would you like to review items before finalizing payment?",
    'recover_password': "Reset your password using this link: [password reset portal]. Check your email for verification.",
    'registration_problems': "Having trouble signing up? Please share the error message you're receiving.",
    'review': "We'd love your feedback! You can leave a product review on any item's details page.",
    'set_up_shipping_address': "Add new addresses in your account's 'Shipping Preferences' section. Multiple addresses supported.",
    'switch_account': "Log out and use different credentials to access another account. Need help switching profiles?",
    'track_order': "Your package is currently: [status]. Full tracking details available here: [tracking link].",
    'track_refund': "Refund status: Processing. You'll receive email confirmation when completed. Estimated 5-7 business days."
}

# Function to clean text input
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text
    return ''

# Function to get chatbot response
def get_response(user_input):
    user_input_cleaned = clean_text(user_input)
    user_input_vectorized = tfidf_vectorizer.transform([user_input_cleaned])
    predicted_intent_index = log_reg_model.predict(user_input_vectorized)[0]
    predicted_intent = label_encoder.inverse_transform([predicted_intent_index])[0]
    response = responses.get(predicted_intent, "Sorry, I didn't understand that.")
    return response

# Streamlit UI
st.title("🤖 Customer Support Chatbot")

# Initialize session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input field
user_input = st.text_input("Type your message here and press Enter:", key="user_input")

# Process user input
if user_input:
    bot_response = get_response(user_input)

    # Append conversation to chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_response))

# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.write(f"**You:** {message}")
    else:
        st.write(f"**Bot:** {message}")
