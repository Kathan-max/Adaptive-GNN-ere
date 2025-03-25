import PyPDF2
import os
import tqdm
import json

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
    return text

def segment_text(text):
    currLen = 0
    chunks = []
    currText = ""
    words = text.split()
    for word in words:
        if currLen + len(word) + (1 if currText else 0) <= 1024:
            currText += " "+word
            currLen += len(word) + 1
        else:
            
            chunks.append(currText)
            currLen = len(word)
            currText = word
    if currText:
        chunks.append(currText) 
    return chunks
        


if __name__ == "__main__":
    # dir = "../Datasets/raw/Law/"
    dir = "../Datasets/raw/Education/"
    # dir = "../Datasets/raw/Finance/"
    text_dict = {}
    for file in tqdm.tqdm(os.listdir(dir), total = len(os.listdir(dir)), desc="Extracting Text"):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(dir, file)
            # print(pdf_path)
            text = extract_text_from_pdf(pdf_path)
            text_dict[file] = text

    # writeFileName = "E:/Windsor/Sem-2/Topics In Applied AI-GNN/Project/Code/data/Datasets/LawFull.json"
    writeFileName = "E:/Windsor/Sem-2/Topics In Applied AI-GNN/Project/Code/data/Datasets/EducationFull.json"
    # writeFileName = "E:/Windsor/Sem-2/Topics In Applied AI-GNN/Project/Code/data/Datasets/FinanceFull.json"
    
    o_chunks = 0
    content = {}
    for key, value in tqdm.tqdm(text_dict.items(), total=len(text_dict), desc="Making the Final Json"):
        textChunks = segment_text(value)
        for i, chunk in enumerate(textChunks):
            content[f"chunk-{o_chunks}"] = {
                "interChunkId": i,
                "fileName": key,
                "text": chunk
            }
            o_chunks += 1
            
    with open(writeFileName, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=4) 