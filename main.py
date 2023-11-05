from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import re
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from nltk.tokenize import sent_tokenize
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# Define the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLInput(BaseModel):
    url: str

class SummarizedText(BaseModel):
    text: str

@app.post("/summarize", response_model=SummarizedText)
async def summarize(url_input: URLInput):
    url = url_input.url
    # Create a fake user agent
    user_agent = UserAgent()

    # Send an HTTP request to the URL with the fake user agent
    headers = {'User-Agent': user_agent.random}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        try:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find and extract text within both <h> (header) and <p> (paragraph) tags
            header_text = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
            paragraph_text = [p.get_text() for p in soup.find_all('p')]

            # Combine all the extracted text into a single string
            all_text = ' '.join(header_text + paragraph_text)

            # Use the regular expression to filter for Arabic text
            arabic_text = re.findall(r'[\u0600-\u06FF\s]+', all_text)

            # Join the extracted Arabic text into a single string
            extracted_arabic_text = ' '.join(arabic_text)

            # Initialize the summarization model
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            m_name = "marefa-nlp/summarization-arabic-english-news"

            tokenizer = AutoTokenizer.from_pretrained(m_name)
            model = AutoModelWithLMHead.from_pretrained(m_name).to(device)

            def get_summary(text, tokenizer, model, device="cpu", num_beams=2):
                if len(text.strip()) < 50:
                    return ["Please provide a longer text"]

                text = "summarize: <paragraph> " + " <paragraph> ".join([s.strip() for s in sent_tokenize(text) if s.strip() != ""]) + " </s>"
                text = text.strip().replace("\n", "")

                tokenized_text = tokenizer.encode(text, return_tensors="pt").to(device)

                summary_ids = model.generate(
                    tokenized_text,
                    max_length=512,
                    num_beams=num_beams,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    early_stopping=True
                )

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                return [s.strip() for s in output.split("<hl>") if s.strip() != ""]

            # Initialize the final resultant text
            final_resultant_text = ""

            # Break the Arabic text into chunks of 1000 characters
            chunk_size = 1000
            chunks = [extracted_arabic_text[i:i + chunk_size] for i in range(0, len(extracted_arabic_text), chunk_size)]

            # Generate and append each chunk's summary to the final resultant text
            for chunk in chunks:
                summaries = get_summary(chunk, tokenizer, model, device)
                for summary in summaries:
                    final_resultant_text += summary + " "

            return {"text": final_resultant_text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Failed to retrieve the page. Status code: {response.status_code}")
    
    
@app.post("/rating", response_model=float)  # Change response_model to float
async def calculate_similarity(url_input: URLInput):
    url = url_input.url

    # Create a fake user agent
    user_agent = UserAgent()

    # Send an HTTP request to the URL with the fake user agent
    headers = {'User-Agent': user_agent.random}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find and extract text within both <h> (header) and <p> (paragraph) tags
        header_text = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
        paragraph_text = [p.get_text() for p in soup.find_all('p')]

        # Combine all the extracted text into a single string
        all_text = ' '.join(header_text + paragraph_text)

        # Use a regular expression to filter for Arabic text
        arabic_text = re.findall(r'[\u0600-\u06FF\s]+', all_text)
        arabic_text = ' '.join(arabic_text)

    else:
        return 0.0  # Return 0.0 for failure to retrieve the page

    # Load keywords from a text file
    file_path = './keywords.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        keywords = file.read()

    # Load a pre-trained Arabic BERT model and tokenizer
    model_name = "aubmindlab/bert-base-arabertv02"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the texts and obtain embeddings
    text1_tokens = tokenizer(arabic_text, return_tensors="pt", padding=True, truncation=True)
    text2_tokens = tokenizer(keywords, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        text1_embeddings = model(**text1_tokens).last_hidden_state.mean(dim=1)
        text2_embeddings = model(**text2_tokens).last_hidden_state.mean(dim=1)

    # Calculate the cosine similarity between the two text embeddings
    similarity = cosine_similarity(text1_embeddings, text2_embeddings)
    similarity_score = similarity[0][0]

    return similarity_score  # Return the similarity score as a float
