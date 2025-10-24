import streamlit as st
import requests

def main():
    st.title("SMS Spam and Ham Classification App")
    
    # User input
    user_input = st.text_input("Enter some data:")

    # Create a button
    if st.button("Check"):
        if user_input.strip() == "": # Check if the input box is empty
            st.warning("Please enter a message!")
        else:
            data = {"message": user_input}
            
            # Hit our API endpoint
            try:
                # Send the message data as a POST request to the Flask backend API
                response = requests.post("http://localhost:5000/predict", json=data)
                if response.status_code == 200:
                    result = response.json()["prediction"]
                    st.success(f"Output: {result}")
                else:
                    st.error("Error occurred in backend")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure Flask server is running.")

if __name__ == "__main__":
    main()
