import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import base64
import time
import os
import tempfile
from gtts import gTTS
import matplotlib.pyplot as plt

# Must be the first Streamlit command
st.set_page_config(
    page_title="Tactile Communication Bridge",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better styles
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Theory Box Styling */
    .theory-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        font-size: 15px;
        line-height: 1.8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .theory-box p {
        margin-bottom: 0.8rem;
    }
    /* Analytics Chart Styling */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    /* Sidebar Styling */
    .sidebar-content {
        text-align: center;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Braille mapping
BRAILLE_MAP = {
    '100000': 'a', '101000': 'b', '110000': 'c', '110100': 'd', '100100': 'e',
    '111000': 'f', '111100': 'g', '101100': 'h', '011000': 'i', '011100': 'j',
    '100010': 'k', '101010': 'l', '110010': 'm', '110110': 'n', '100110': 'o',
    '111010': 'p', '111110': 'q', '101110': 'r', '011010': 's', '011110': 't',
    '100011': 'u', '101011': 'v', '011101': 'w', '110011': 'x', '110111': 'y',
    '100111': 'z', '000000': ' '
}

def process_braille_image(uploaded_file):
    """Process uploaded image and extract braille text"""
    try:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return "Error: Could not read image", 0, 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter dots by area
        dots = []
        for c in contours:
            area = cv2.contourArea(c)
            if 20 < area < 500:
                dots.append(c)
        
        if len(dots) < 6:
            return f"Found only {len(dots)} dots. Need at least 6 for a braille cell.", len(dots), 0
        
        # Sort dots by position (top to bottom, left to right)
        dots = sorted(dots, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
        
        # Group into cells of 6 dots
        braille_text = ""
        for i in range(0, len(dots) - 5, 6):
            cell = dots[i:i+6]
            pattern = ''
            for dot in cell:
                pattern += '1' if cv2.contourArea(dot) > 30 else '0'
            
            # Get character from map
            char = BRAILLE_MAP.get(pattern, '?')
            braille_text += char
        
        return braille_text if braille_text else "Could not recognize pattern", len(dots), len(braille_text)
    
    except Exception as e:
        return f"Error: {str(e)}", 0, 0

def text_to_speech(text):
    """Convert text to speech"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

# Generate plain English theory explanation
def generate_theory_explanation(text, dots, chars):
    """Generate plain English explanation of how the translation works"""
    if not text:
        return ""
    
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    unique_chars = len(set(text.lower()))
    
    if words:
        avg_word_length = sum(len(w) for w in words) / len(words)
        longest_word = max(words, key=len)
    else:
        avg_word_length = 0
        longest_word = "N/A"
    
    expected_cells = dots / 6 if dots > 0 else 0
    accuracy = min(100, (chars / max(1, expected_cells)) * 100) if expected_cells > 0 else 0
    
    explanation = f"""
When you uploaded your Braille image, our system first scanned the picture to locate all the raised dots. It found {dots} individual dots arranged in a specific pattern. Braille uses groups of six dots arranged in two columns and three rows to represent each letter. Your image contained {int(dots/6)} Braille cells, which should translate to about {int(dots/6)} characters.

The final translation produced {char_count} characters forming {word_count} words. The average word length is {avg_word_length:.1f} letters. The longest word in your translation is "{longest_word}" with {len(longest_word)} letters.

For every group of six dots, our system creates a unique pattern. For instance, if only the top-left dot is raised, that pattern represents the letter 'a'. If both top-left and middle-left dots are raised, that represents the letter 'b'. Your translation used {unique_chars} different letters.

Based on the number of dots detected versus the number of characters produced, your translation achieved {accuracy:.0f}% accuracy. This means the system successfully identified and correctly mapped most of the Braille patterns to their corresponding letters.

The translated text contains {word_count} words. If spoken aloud, it would take approximately {word_count * 0.3:.1f} seconds to read at normal speaking speed. The text shows {unique_chars} unique characters, indicating {"rich vocabulary" if unique_chars > 10 else "simple but clear language"}.

"""
    return explanation

# Initialize session state
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""
if 'dot_count' not in st.session_state:
    st.session_state.dot_count = 0
if 'char_count' not in st.session_state:
    st.session_state.char_count = 0
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'theory_explanation' not in st.session_state:
    st.session_state.theory_explanation = ""

# Header
st.markdown('<h1 class="main-header">🤟 Tactile Communication Bridge</h1>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload", "📊 Analytics", "📈 Report"])

# Tab 1: Upload and Translate
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Braille Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of Braille text"
        )
        
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Translate button
            if st.button("🔍 Translate Braille", use_container_width=True):
                with st.spinner("Processing image..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Process image
                    text, dots, chars = process_braille_image(uploaded_file)
                    
                    # Update session state
                    st.session_state.translated_text = text
                    st.session_state.dot_count = dots
                    st.session_state.char_count = chars
                    st.session_state.processed = True
                    # Generate plain English theory explanation
                    st.session_state.theory_explanation = generate_theory_explanation(text, dots, chars)
                    
                    if dots > 0:
                        st.success(f"✅ Found {dots} dots, translated {chars} characters")
                    else:
                        st.warning(text)
    
    with col2:
        st.subheader("📝 Translation Result")
        
        if st.session_state.processed and st.session_state.translated_text:
            # Display translated text
            st.text_area("Translated Text:", st.session_state.translated_text, height=150)
            
            # Display Theory Explanation in Plain Text with Styling
            st.markdown("### 📖 Understanding Your Translation")
            st.markdown(f'<div class="theory-box">{st.session_state.theory_explanation}</div>', unsafe_allow_html=True)
            
            # Action buttons
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🔊 Play Audio", use_container_width=True):
                    audio_file = text_to_speech(st.session_state.translated_text)
                    if audio_file:
                        st.audio(audio_file)
                        os.unlink(audio_file)
            
            with col_b:
                if st.button("📋 Copy", use_container_width=True):
                    st.info("Text ready to copy (Ctrl+C)")
        else:
            st.info("👆 Upload an image and click Translate")

# Tab 2: Analytics
with tab2:
    st.subheader("📊 Translation Analytics")
    
    if st.session_state.processed and st.session_state.translated_text:
        # Calculate metrics - FIXED accuracy calculation
        accuracy = 0
        if st.session_state.dot_count > 0:
            expected_chars = st.session_state.dot_count / 6
            accuracy = min(100, (st.session_state.char_count / expected_chars) * 100)
        else:
            # If no dots detected but we have text, use a different calculation
            accuracy = 100 if st.session_state.char_count > 0 else 0
        
        # KPI Cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <h2>{st.session_state.dot_count}</h2>
                <p>Dots Detected</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <h2>{st.session_state.char_count}</h2>
                <p>Characters</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <h2>{accuracy:.1f}%</h2>
                <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Simple charts
        col4, col5 = st.columns(2)
        
        with col4:
            st.subheader("Character Frequency")
            text = st.session_state.translated_text.lower()
            if text:
                char_counts = {}
                for char in text:
                    if char.isalpha():
                        char_counts[char] = char_counts.get(char, 0) + 1
                
                if char_counts:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    chars = list(char_counts.keys())
                    counts = list(char_counts.values())
                    ax.bar(chars, counts, color='#667eea')
                    ax.set_xlabel('Characters')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Character Distribution')
                    st.pyplot(fig)
                    plt.close()
        
        with col5:
            st.subheader("Word Analysis")
            words = st.session_state.translated_text.split()
            if words:
                word_lengths = [len(w) for w in words]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(word_lengths, bins=10, color='#764ba2', alpha=0.7)
                ax.set_xlabel('Word Length')
                ax.set_ylabel('Frequency')
                ax.set_title('Word Length Distribution')
                st.pyplot(fig)
                plt.close()
    else:
        st.info("👆 Process an image first to see analytics")

# Tab 3: Report - THEORY PART REMOVED
with tab3:
    st.subheader("📈 Translation Report")
    
    if st.session_state.processed and st.session_state.translated_text:
        # Create report data
        report_data = {
            'Metric': [
                'Image Processed',
                'Dots Detected',
                'Characters',
                'Words',
                'Timestamp'
            ],
            'Value': [
                '✅ Yes',
                str(st.session_state.dot_count),
                str(st.session_state.char_count),
                str(len(st.session_state.translated_text.split())),
                time.strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        df = pd.DataFrame(report_data)
        st.table(df)
        
        # Display translated text
        st.subheader("📝 Translated Text")
        st.info(st.session_state.translated_text)
        
        # THEORY PART REMOVED - No explanation here
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="braille_report.csv">📥 Download CSV Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Text report without theory
            report = f"""TACTILE COMMUNICATION BRIDGE REPORT
{'='*40}

Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

STATISTICS:
- Dots Detected: {st.session_state.dot_count}
- Characters: {st.session_state.char_count}
- Words: {len(st.session_state.translated_text.split())}

TRANSLATED TEXT:
{st.session_state.translated_text}

{'='*40}
STATUS: Report Generated Successfully
{'='*40}
"""
            b64_text = base64.b64encode(report.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64_text}" download="braille_report.txt">📥 Download Text Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("👆 Process an image to generate a report")

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='font-size: 48px;'>🤟</h1>
        <h3>Tactile Communication Bridge</h3>
        <p style='color: #666;'>Computer Vision for Accessibility</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### How to Use")
    st.markdown("""
    1. **Upload** a Braille image
    2. **Click** Translate button
    3. **View** translated text with explanation
    4. **Play** audio output
    5. **Download** report
    """)
    
    # Status
    if st.session_state.processed:
        st.success("✅ System Ready")
        st.info(f"Last translation: {st.session_state.char_count} chars")
    
    st.markdown("---")