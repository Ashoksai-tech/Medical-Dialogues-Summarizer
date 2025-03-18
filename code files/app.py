# Streamlit Web Application for Physician Notetaker
import streamlit as st
from transformers import  BertForTokenClassification, BertTokenizerFast, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import zipfile
import shutil

# Load the fine-tuned NER model and tokenizer
@st.cache_resource
def load_ner_model():
    # Where the unzipped model files will be stored
    model_path = "./physician_notetaker_model"
    
    # Path to your downloaded .zip file (replace with your actual path)
    zip_path = r"C:\\Users\\tws\Downloads\\physician_notetaker_model (1).zip"  # Updated from screenshot
    
    # If model folder doesn’t exist but zip does, unzip it
    if not os.path.exists(model_path) and os.path.exists(zip_path):
        # Create the model_path directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        # Unzip directly into model_path
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_path)
        st.success(f"Unzipped {zip_path} to {model_path}")
    
    if not os.path.exists(model_path):
        st.error(f"Model directory {model_path} not found. Please ensure {zip_path} is present and contains the correct model files.")
        return None
    
    # Verify required files exist
    required_files = ["config.json", "model.safetensors", "vocab.txt"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        st.error(f"Missing required model files: {missing_files}")
        return None
    
    ner_model = BertForTokenClassification.from_pretrained(model_path)
    ner_tokenizer = BertTokenizerFast.from_pretrained(model_path)
    return pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

# Load other pipelines
@st.cache_resource
def load_pipelines():
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        sentiment_analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
        zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        return summarizer, sentiment_analyzer, zero_shot
    except Exception as e:
        st.error(f"Error loading pipelines: {e}")
        return None, None, None

# Process transcript function
def process_transcript(transcript, ner_pipeline, summarizer, sentiment_analyzer, zero_shot):
    intents = ["Seeking reassurance", "Reporting symptoms", "Expressing concern"]
    
    # NER
    ner_results = ner_pipeline(transcript)
    entity_dict = {"Symptoms": [], "Diagnosis": [], "Treatments": [], "Prognosis": []}
    for entity in ner_results:
        label = entity["entity_group"]
        text = entity["word"]
        if "SYMPTOM" in label:
            entity_dict["Symptoms"].append(text)
        elif "DIAGNOSIS" in label:
            entity_dict["Diagnosis"].append(text)
        elif "TREATMENT" in label:
            entity_dict["Treatments"].append(text)
        elif "PROGNOSIS" in label:
            entity_dict["Prognosis"].append(text)

    # Summary and keywords
    cleaned = " ".join(transcript.split("\n"))
    summary = summarizer(cleaned, max_length=50, min_length=20)[0]["summary_text"]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5)
    tfidf_matrix = vectorizer.fit_transform([cleaned])
    keywords = vectorizer.get_feature_names_out().tolist()

    medical_details = {
        "Patient_Name": "Unknown" if "Mr." not in transcript and "Ms." not in transcript else transcript.split("Mr." if "Mr." in transcript else "Ms.")[1].split()[0],
        "Symptoms": list(set(entity_dict["Symptoms"])),
        "Diagnosis": list(set(entity_dict["Diagnosis"])),
        "Treatments": list(set(entity_dict["Treatments"])),
        "Current_Status": "Occasional symptoms" if "occasional" in cleaned.lower() else "Improving",
        "Prognosis": list(set(entity_dict["Prognosis"])) or ["Full recovery expected"]
    }

    # Sentiment and Intent
    patient_lines = [line.split("Patient:")[1].strip() for line in transcript.split("\n") if "Patient:" in line]
    sentiment_intent = []
    for line in patient_lines:
        sentiment_result = sentiment_analyzer(line)[0]
        emotion = sentiment_result["label"]
        score = sentiment_result["score"]
        sentiment = "Anxious" if emotion in ["fear", "sadness"] and score > 0.7 else "Reassured" if emotion in ["joy", "love"] and score > 0.7 else "Neutral"
        intent_result = zero_shot(line, intents)
        intent = intent_result["labels"][0]
        sentiment_intent.append({"Line": f"Patient: {line}", "Sentiment": sentiment, "Intent": intent})

    # SOAP Note
    soap = {
        "Subjective": {
            "Chief_Complaint": " and ".join(medical_details["Symptoms"]) or "Not specified",
            "History_of_Present_Illness": summary
        },
        "Objective": {
            "Physical_Exam": next((line.split("Physician:")[1].strip() for line in transcript.split("\n") if "examination" in line.lower() or "check" in line.lower() or "vitals" in line.lower()), "Not recorded"),
            "Observations": "Patient appears stable."
        },
        "Assessment": {
            "Diagnosis": medical_details["Diagnosis"] or ["Pending evaluation"],
            "Severity": "Mild, improving" if "better" in cleaned.lower() else "Under evaluation"
        },
        "Plan": {
            "Treatment": medical_details["Treatments"] or ["Monitor and manage symptoms"],
            "Follow-Up": "Return if symptoms worsen or persist beyond six months."
        }
    }

    return medical_details, sentiment_intent, soap

# Streamlit app
def main():
    st.title("Physician Notetaker Web App")
    st.write("Enter a medical transcript below to analyze it using a fine-tuned BioBERT model.")

    # Load models
    ner_pipeline = load_ner_model()
    if ner_pipeline is None:
        return
    summarizer, sentiment_analyzer, zero_shot = load_pipelines()
    if summarizer is None:
        return

    # Text input
    transcript = st.text_area("Enter Transcript", height=300, value="""Physician: Hello, Mr. Brown. How are you today?
Patient: Hi, doctor. I’ve been having headaches and feeling tired for a week.
Physician: Have you tried anything for it?
Patient: Yes, I took some aspirin, but it only helps a little.
Physician: Let’s examine you. [Pause] Your vitals are normal, but it could be stress-related.
Physician: I recommend rest and hydration; you should feel better soon.""")

    if st.button("Analyze Transcript"):
        if transcript.strip():
            with st.spinner("Processing..."):
                try:
                    med_details, sent_intent, soap = process_transcript(transcript, ner_pipeline, summarizer, sentiment_analyzer, zero_shot)
                    
                    # Display results
                    st.subheader("Medical Details")
                    st.json(med_details)
                    
                    st.subheader("Sentiment & Intent")
                    for item in sent_intent:
                        st.json(item)
                    
                    st.subheader("SOAP Note")
                    st.json(soap)
                except Exception as e:
                    st.error(f"Error processing transcript: {e}")
        else:
            st.warning("Please enter a valid transcript.")

if __name__ == "__main__":
    main()