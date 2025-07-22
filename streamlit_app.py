import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Lyric Generator",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the model and tokenizer (cached for performance)"""
    try:
        # Load original GPT-2 tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load fine-tuned model from local directory
        model_path = "./model"  # Assumes model files are in ./model/ directory
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the model files are in the './model/' directory")
        return None, None, None

def generate_lyrics(tokenizer, model, device, prompt, **generation_params):
    """Generate lyrics for a single prompt"""
    encoded_prompt = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=encoded_prompt,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            **generation_params
        )
    
    generated_texts = []
    for sequence in output_sequences:
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts

def create_download_content(all_results):
    """Create formatted text content for download"""
    content = "GENERATED LYRICS\n"
    content += "=" * 50 + "\n"
    content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    content += "=" * 50 + "\n\n"
    
    for prompt_idx, (prompt, generations) in enumerate(all_results, 1):
        content += f"PROMPT {prompt_idx}: {prompt}\n"
        content += "-" * 60 + "\n\n"
        
        for gen_idx, text in enumerate(generations, 1):
            content += f"--- Generation {gen_idx} ---\n\n"
            content += text + "\n\n"
            content += "-" * 40 + "\n\n"
        
        content += "\n"
    
    return content

# Main app
def main():
    st.title("Lyric Generator")
    st.markdown("*Generate \"creative\" lyrics using a fine-tuned GPT-2 model*")

    # Load model
    tokenizer, model, device = load_model()
    
    if tokenizer is None or model is None:
        st.stop()
    
    st.success("Model loaded successfully.")
    
    # Sidebar for parameters
    st.sidebar.header("Generation Parameters")
    
    # Generation parameters with sliders
    num_sequences_per_prompt = st.sidebar.number_input(
        "Number of lyrics",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Number of lyrics to generate"
    )
    
    min_length = st.sidebar.slider(
        "Minimum length",
        min_value=20,
        max_value=200,
        value=80,
        step=1,
        help="Minimum number of tokens to generate"
    )
    
    max_length = st.sidebar.slider(
        "Maximum length",
        min_value=100,
        max_value=500,
        value=300,
        step=1,
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.2,
        step=0.01,
        help="Controls randomness: higher = more random"
    )
    
    top_p = st.sidebar.slider(
        "Top-p (nucleus sampling)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.01,
        help="Probability threshold for token selection"
    )
    
    top_k = st.sidebar.slider(
        "Top-k sampling",
        min_value=10,
        max_value=100,
        value=50,
        step=1,
        help="Number of top tokens to consider"
    )
    
    repetition_penalty = st.sidebar.slider(
        "Repetition penalty",
        min_value=0.5,
        max_value=2.0,
        value=1.1,
        step=0.01,
        help="Penalty for repeating tokens"
    )
    
    # Main content area
    st.header("Enter a first line, word, or phrase")

    # Single prompt input
    prompt = st.text_input(
        "Lyric prompt:",
        value="Early one morning the sun was shining",
        placeholder="Enter a starting phrase for your lyrics..."
    )

    # Generate button
    if st.button("Generate Lyrics", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt")
            return
        
        # Progress indicator
        status_text = st.empty()
        status_text.text(f"Generating lyrics for: '{prompt}'")
        
        try:
            generated_texts = generate_lyrics(
                tokenizer, model, device, prompt,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_sequences_per_prompt
            )
            
            status_text.text("Generation complete")
            
        except Exception as e:
            st.error(f"Error generating lyrics: {e}")
            return
        


        # Display results
        st.header("Generated Lyrics")
        st.subheader(f"Prompt: '{prompt}'")
        
        # Create tabs for each generation
        tab_names = [f"Generation {i+1}" for i in range(len(generated_texts))]
        tabs = st.tabs(tab_names)
        
        for tab_idx, (tab, text) in enumerate(zip(tabs, generated_texts)):
            with tab:
                st.text_area(
                    f"Generated lyrics {tab_idx + 1}:",
                    value=text,
                    height=300,
                    key=f"result_{tab_idx}"
                )
        
        # Download button
        download_content = create_download_content([(prompt, generated_texts)])
        st.download_button(
            label="Download Lyrics as Text File",
            data=download_content,
            file_name=f"generated_lyrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.markdown("The model can take a little while to generate. Try a higher number for 'Number of lyrics' to generate more results for each run.")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and Transformers*")

if __name__ == "__main__":
    main()