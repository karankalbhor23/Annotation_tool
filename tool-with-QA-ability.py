import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


st.set_page_config(page_title="Image Description & Q&A Generator", page_icon="📝", layout="centered")


model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    load_in_4bit=True,
    _attn_implementation='eager'
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)


st.markdown("""
    <style>
        .title {
            font-size: 36px;
            color: #4A90E2;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 18px;
            color: #555555;
            text-align: center;
            margin-bottom: 10px;
        }
        .caption {
            font-size: 16px;
            color: #4A4A4A;
            text-align: center;
            margin-top: 20px;
            font-style: italic;
        }
        .footer {
            text-align: center;
            font-size: 12px;
            color: #888888;
            margin-top: 50px;
        }
        .stButton>button {
            color: white;
            background-color: #4A90E2;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<div class='title'>Image Description & Q&A Generator📝</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image, get a description, and ask questions about it</div>", unsafe_allow_html=True)


uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    
    st.write("### Uploaded Image:")
    st.image(uploaded_image, width=400, use_column_width=True)

    
    image = Image.open(uploaded_image)

    
    placeholder = "<|image_1|>"

    
    description_prompt = [
        {"role": "user", "content": placeholder + """Generate a meticulously detailed, single-paragraph description of the image for Vision Language Model (Vision LLM) training, capturing all visually observable elements while avoiding assumptions or interpretations beyond what is visible. Adhere strictly to the following criteria:

Prohibited Elements:
No assumptions or inferences beyond visual data.
Avoid hallucinations, invented details, or subjective perspectives.
Do not include speculative language, emotional inferences, or unverifiable information.
Exclude contextual associations unrelated to visible elements.
Required Descriptions:
Visual Entities: Describe each visible object, including color, texture, size, and shape.
Spatial Relationships: Indicate the precise positioning and proximity of objects.
Visible Text: Extract and report any text present (e.g., signs, labels, or logos) with font style, color, and size if discernible.
Color Palette: Analyze primary, secondary, and contrasting colors.
Texture Analysis: Describe textures such as smooth, rough, or shiny where applicable.
Composition: Detail the layout, arrangement, and balance of elements.
Object Categories and Scene Classification: Classify objects based on visual characteristics and categorize the scene (e.g., urban, rural, indoor).
Output Guidelines:
Length: Between 250-300 words in a single, grammatically correct paragraph.
Clarity: Ensure factual accuracy, concise language, and precise terminology without ambiguous language or jargon.
Focus: Center solely on visually observable details, following these guidelines to optimize accuracy and reliability for Vision LLM training.
Example Output:
'The image shows a busy urban street in bright daylight. A 4-story beige building with “Bistro Cafe” in red lettering dominates the left side. Cars, primarily sedans and compact vehicles, are aligned on the street, which runs from the bottom to the top of the frame. A large, blue billboard displays "SALE 50% OFF," with bold white text, visible on the upper-right corner. The color palette features shades of gray, beige, blue, and red, with textures from the smooth car surfaces to rough building exteriors. Scene classification: urban street setting with visible objects, including buildings, vehicles, and street signs.'

Explanation of Example Output:
This example illustrates a structured and precise description, covering each required detail without assumptions:

Visual Entities: Each major object is specified—the beige building, cars, and a blue billboard. The color, shape, and placement of these entities are clearly stated.
Spatial Relationships: The building is positioned on the left side, cars are aligned along the street, and the billboard is noted in the upper-right corner. These details create a clear picture of how objects are arranged within the scene.
Visible Text: The text on the building and billboard, “Bistro Cafe” and “SALE 50% OFF,” is recorded accurately, including font color and style, contributing to factual precision.
Color Palette: The colors (gray, beige, blue, red) are analyzed and presented without additional interpretation, covering primary and secondary colors to enhance visual accuracy.
Texture Analysis: Textures such as the smooth surfaces of cars and the rough exteriors of buildings are described, adding another layer of visual detail.
Composition: The layout is discussed, noting the street’s orientation and the placement of objects to capture the scene's organization and balance.
Object Categories and Scene Classification: The description classifies the setting as an “urban street,” which provides clear context without additional, unverifiable details."""}
    ]

    
    prompt = processor.tokenizer.apply_chat_template(description_prompt, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

   
    generation_args = {
        "max_new_tokens": 2000,
        "temperature": 1.5,
        "do_sample": False,
    }
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    description = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    
    st.write("### Description:")
    st.write(description)

    
    st.markdown("### Ask Questions about the Image:")
    user_question = st.text_input("Type your question here...")

    if user_question:
        
        question_prompt = [
            {"role": "user", "content": placeholder + user_question}
        ]
        prompt = processor.tokenizer.apply_chat_template(question_prompt, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

        
        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        answer = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        
        st.write("**Answer:**")
        st.write(answer)


st.markdown("<div class='footer'>Developed by karankalbhor23</div>", unsafe_allow_html=True)
