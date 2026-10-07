import sys
import asyncio

# Silence Windows asyncio socket cleanup warnings
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Configure Page Config FIRST before any other streamlit calls
st.set_page_config(
    page_title="Hungarian to English NMT Comparison",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Hungarian → English Translator: Attention vs. Without Attention")
st.markdown("##### *Compare Seq2Seq Model Performance with and without Luong Attention*")
st.divider()

# Cache heavy model loading and initialization for both pipelines
@st.cache_resource
def load_translators():
    import pickle
    import tensorflow as tf
    from src.translate_withoutattn import translate_without_attention
    from src.translate_attention import translate_with_attention
    from src.model_attention import Encoder, Decoder
    
    # Load tokenizers
    with open("models/source_tokenizer.pkl", "rb") as f:
        source_tokenizer = pickle.load(f)
    with open("models/target_tokenizer.pkl", "rb") as f:
        target_tokenizer = pickle.load(f)
        
    # Load max encoding length from metadata if available, otherwise default to 20
    try:
        with open("models/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            max_encoding_len = metadata.get("max_len_src", 20)
    except:
        max_encoding_len = 20
        
    # Re-initialize the attention model architecture
    embedding_dim = 256
    hidden_dim = 512
    num_encoder_tokens = len(source_tokenizer.word_index) + 1
    num_decoder_tokens = len(target_tokenizer.word_index) + 1
    
    encoder = Encoder(num_encoder_tokens, embedding_dim, hidden_dim)
    decoder = Decoder(num_decoder_tokens, embedding_dim, hidden_dim)
    
    # BUILD MODELS BY RUNNING DUMMY INPUTS (Fixes subclassed variable initialization error)
    dummy_src = tf.zeros((1, max_encoding_len), dtype=tf.int32)
    enc_out, state_h, state_c = encoder(dummy_src)
    dummy_tgt = tf.zeros((1, 1), dtype=tf.int32)
    decoder((dummy_tgt, enc_out, (state_h, state_c)))
    
    # Load trained weights safely now that variables exist
    encoder.load_weights("models/encoder_attn_weights.h5")
    decoder.load_weights("models/decoder_attn_weights.h5")
    
    # Wrapper function matching your translate_with_attention signature
    def wrapped_translate_attention(sentence):
        return translate_with_attention(
            sentence, 
            source_tokenizer=source_tokenizer, 
            encoder=encoder, 
            target_tokenizer=target_tokenizer, 
            decoder=decoder, 
            max_encoding_len=max_encoding_len
        )

    return translate_without_attention, wrapped_translate_attention

try:
    with st.spinner("Loading both translation models into memory..."):
        translate_no_attn, translate_attn = load_translators()
    
    # Input UI
    user_sentence = st.text_area(
        "Enter Hungarian text to translate:",
        placeholder="e.g., Nem számítasz nekem.",
        height=80
    )

    # Optional Reference sentence for BLEU score comparison
    reference_sentence = st.text_input(
        "Optional: Enter English Reference Sentence (for BLEU score evaluation):",
        placeholder="e.g., You don't matter to me."
    )

    # Translate Action
    if st.button("Compare Translations", type="primary", use_container_width=True):
        if user_sentence.strip():
            with st.spinner("Translating with both models..."):
                # 1. Without Attention Translation
                tokenized_input_no, translation_no = translate_no_attn(user_sentence)
                
                # 2. With Attention Translation
                tokenized_input_att, translation_att = translate_attn(user_sentence)
                
                st.subheader("Model Comparison Results")
                col1, col2 = st.columns(2)

                # Without Attention Column
                with col1:
                    st.info("**Without Attention Model**")
                    st.write(f"**Translation:** {translation_no}")
                    
                    if reference_sentence.strip():
                        ref_tokens = reference_sentence.lower().split()
                        hyp_tokens = translation_no.lower().split()
                        smoothing = SmoothingFunction().method1
                        bleu_no = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
                        st.metric(label="BLEU Score", value=f"{bleu_no:.4f}")

                # With Attention Column
                with col2:
                    st.success("**Luong Attention Model**")
                    st.write(f"**Translation:** {translation_att}")
                    
                    if reference_sentence.strip():
                        ref_tokens = reference_sentence.lower().split()
                        hyp_tokens = translation_att.lower().split()
                        smoothing = SmoothingFunction().method1
                        bleu_att = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
                        delta_val = f"{bleu_att - bleu_no:.4f}"
                        st.metric(label="BLEU Score", value=f"{bleu_att:.4f}", delta=delta_val)

                # Tokenized view expander
                with st.expander("View Tokenized Input Details"):
                    st.write("Tokenized Sequence:", tokenized_input_no[0] if tokenized_input_no else "")
        else:
            st.warning("Please enter a sentence to translate.")

except Exception as e:
    st.error(f"Failed to load translation engines: {str(e)}")