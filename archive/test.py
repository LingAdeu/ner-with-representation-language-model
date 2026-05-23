# import streamlit as st
# from st_annotator import text_annotator

# st.title("Interactive NER Editor")

# # Correct format: A dictionary with a category key containing the list of spans
# initial_labels = {
#     "NER": [
#         {"start": 0, "end": 5, "label": "ORG", "text": "Apple"},
#         {"start": 40, "end": 53, "label": "GPE", "text": "San Francisco"}
#     ]
# }

# raw_text = "Apple is looking at buying a startup in San Francisco."

# # Pass the dictionary to the 'labels' parameter
# result = text_annotator(
#     text=raw_text,
#     labels=initial_labels,
#     key="ner_editor"
# )

# st.subheader("Model Data Output (JSON)")
# st.write(result)

import streamlit as st
from st_annotator import text_annotator

st.title("Interactive NER Editor")

raw_text = "Apple is looking at buying a startup in San Francisco."

# Pre-populate with real annotations + dummy seed for each desired label
initial_annotations = {
    "NER": [
        # real annotations
        {"start": 0,  "end": 5,  "label": "ORG", "text": "Apple"},
        {"start": 40, "end": 53, "label": "GPE", "text": "San Francisco"},
        # dummy “seeds” – they add no visible text but teach the component the labels
        {"start": 0, "end": 0, "label": "PERSON", "text": ""},
        {"start": 0, "end": 0, "label": "DATE",   "text": ""},
    ]
}

result = text_annotator(
    text=raw_text,
    labels=initial_annotations,
    key="ner_editor"
)

st.subheader("Edited Annotations")
st.json(result)