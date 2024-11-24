import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import os
from typing import Tuple

# Configure API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key="YOUR_API_KEY")

def initialize_model():
    """Initialize the gemini-1.5-flash"""
    return genai.GenerativeModel('gemini-1.5-flash')

def process_uploaded_image(uploaded_file) -> Tuple[Image.Image, str]:
    """Process the uploaded image and convert to MIME type"""
    image = Image.open(uploaded_file)
    
    # Convert to RGB if necessary
    if image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')
    
    # Create byte stream
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='JPEG')
    byte_array = byte_stream.getvalue()
    
    return image, byte_array

def get_object_detection_prompt():
    """Returns the prompt for object detection and safety analysis"""
    return """
    Analyze this image for safety and navigation purposes. Please provide:
    1. A list of all visible objects and potential obstacles
    2. Their approximate locations in the scene
    3. Any potential safety hazards or concerns
    4. Specific navigation guidance considering the identified objects
    
    Format your response in clear sections using markdown.
    """

def get_daily_tasks_prompt():
    """Returns the prompt for daily task assistance"""
    return """
    Analyze this image to provide assistance with daily tasks. Please:
    1. Identify any text or labels visible in the image
    2. Describe the items and their potential uses
    3. Provide relevant context-specific information
    4. Suggest any helpful tips for interacting with the identified items
    
    Format your response in clear sections using markdown.
    """

def get_ai_analysis(image, prompt, model):
    """Get AI analysis using Gemini Pro Vision"""
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def main():
    st.set_page_config(
        page_title="AI Vision Assistant",
        page_icon="👁️",
        layout="wide"
    )
    
    # Header
    st.title("👁️ AI Vision Assistant")
    st.write("""
    Upload an image to receive AI-powered assistance with:
    * Object and obstacle detection for safe navigation
    * Daily task guidance and item recognition
    """)
    
    try:
        model = initialize_model()
    except Exception as e:
        st.error(f"Error initializing Gemini API: {str(e)}")
        st.info("Please ensure you have set your GOOGLE_API_KEY environment variable.")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            # Process image
            image, image_bytes = process_uploaded_image(uploaded_file)
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_column_width=True)
            
            # Analysis options
            with col2:
                st.subheader("Analysis Options")
                analysis_type = st.radio(
                    "Choose analysis type:",
                    ["Safety & Navigation", "Daily Tasks Assistance"]
                )
                
                if st.button("Analyze Image"):
                    with st.spinner("Analyzing image..."):
                        if analysis_type == "Safety & Navigation":
                            prompt = get_object_detection_prompt()
                            analysis = get_ai_analysis(image, prompt, model)
                            
                            st.subheader("Safety & Navigation Analysis")
                            st.markdown(analysis)
                            
                        else:  # Daily Tasks Assistance
                            prompt = get_daily_tasks_prompt()
                            analysis = get_ai_analysis(image, prompt, model)
                            
                            st.subheader("Daily Tasks Assistance")
                            st.markdown(analysis)
                        
                        # Add helpful tips based on analysis type
                        st.info(
                            "💡 Tip: "
                            + ("Take note of any highlighted hazards and follow the navigation guidance carefully."
                               if analysis_type == "Safety & Navigation"
                               else "Save or bookmark this analysis for future reference when performing similar tasks.")
                        )
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please try uploading a different image or try again later.")

    # Add instructions in sidebar
    with st.sidebar:
        st.subheader("How to Use")
        st.write("""
        1. Upload an image using the file uploader
        2. Choose the type of analysis you need:
           * Safety & Navigation - For understanding surroundings and obstacles
           * Daily Tasks - For help with identifying items and reading labels
        3. Click 'Analyze Image' to get AI-powered assistance
        """)
        
        st.subheader("About")
        st.write("""
        This application uses Google's Gemini-1.5-flash AI to provide:
        * Object and obstacle detection
        * Text recognition and reading
        * Context-specific guidance
        * Safety awareness
        """)

if __name__ == "__main__":
    main()
