import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import spacy
from sklearn.manifold import TSNE
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_text_from_ocr(directory_path: str) -> dict:
    ocr_texts = {}
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                ocr_texts[filename[:-4]] = f.read()
    return ocr_texts


def run_spacy_ner(text_data: dict):
    print("\n--- Running spaCy Named Entity Recognition ---")
    try:
        nlp = spacy.load("de_core_news_md")
    except OSError:
        print("SpaCy model 'de_core_news_md' not found. Please run: python -m spacy download de_core_news_md")
        return

    all_word_vectors = []
    entity_vectors_by_type = defaultdict(list)
    word_labels = []
    entity_texts_by_type = defaultdict(list)

    for doc_name, text in text_data.items():
        doc = nlp(text)
        print(f"\nProcessing document: {doc_name}")

        print("Entities found:")
        for ent in doc.ents:
            print(f"  - {ent.text} ({ent.label_})")
            if ent.has_vector:
                entity_vectors_by_type[ent.label_].append(ent.vector)
                entity_texts_by_type[ent.label_].append(ent.text)
        
        for token in doc:
            if token.has_vector and not token.is_punct and not token.is_space and token.text.strip() != "":
                all_word_vectors.append(token.vector)
                word_labels.append(token.text)


        print("\nSimilar words (examples):")
        target_words = ["Standort", "Datum", "Fotograf", "Museum"]
        for word_str in target_words:
            if word_str in nlp.vocab and nlp.vocab[word_str].has_vector:
                ms = nlp.vocab.vectors.most_similar(np.asarray([nlp.vocab[word_str].vector]), n=5)
                similar_words = [nlp.vocab.strings[w_id] for w_id in ms[0][0]]
                print(f"  - Words similar to '{word_str}': {', '.join(similar_words)}")
            else:
                print(f"  - '{word_str}' not found or has no vector in vocabulary for similarity search.")


    if all_word_vectors and len(all_word_vectors) > 1:
        tsne_model_all = TSNE(n_components=2, random_state=0, perplexity=min(30, len(all_word_vectors) - 1))
        np.seterr(all='ignore')
        all_word_vectors_array = np.array(all_word_vectors)
        all_word_vectors_array = all_word_vectors_array[~np.isnan(all_word_vectors_array).any(axis=1) & ~np.isinf(all_word_vectors_array).any(axis=1)]
        
        if all_word_vectors_array.shape[0] > 1:
            tsne_all_coords = tsne_model_all.fit_transform(all_word_vectors_array)
            print("\nPrepared word vectors for t-SNE visualization.")
            
            plt.figure(figsize=(12, 10))
            plt.scatter(tsne_all_coords[:, 0], tsne_all_coords[:, 1], s=10)
            plt.title("t-SNE of All Word Vectors")
            
            step = max(1, len(tsne_all_coords) // 50)
            for i, (x, y) in enumerate(tsne_all_coords):
                if i % step == 0:
                    plt.annotate(word_labels[i], (x, y), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)
            
            plt.grid(True)
            plt.show()
        else:
            print("Not enough valid word vectors for t-SNE visualization after filtering.")
    else:
        print("No word vectors found for t-SNE visualization.")

    for ent_type, vectors in entity_vectors_by_type.items():
        if vectors and len(vectors) > 1:
            perplexity_val = min(30, len(vectors) - 1)
            if perplexity_val < 1:
                print(f"Not enough samples for t-SNE of '{ent_type}' entity vectors.")
                continue

            tsne_model_ent = TSNE(n_components=2, random_state=0, perplexity=perplexity_val)
            vectors_array = np.array(vectors)
            vectors_array = vectors_array[~np.isnan(vectors_array).any(axis=1) & ~np.isinf(vectors_array).any(axis=1)]

            if vectors_array.shape[0] > 1:
                tsne_ent_coords = tsne_model_ent.fit_transform(vectors_array)
                print(f"Prepared '{ent_type}' entity vectors for t-SNE visualization.")
                
                plt.figure(figsize=(10, 8))
                plt.scatter(tsne_ent_coords[:, 0], tsne_ent_coords[:, 1], s=20)
                plt.title(f"t-SNE of {ent_type} Entity Vectors")
                
                for i, (x, y) in enumerate(tsne_ent_coords):
                    plt.annotate(entity_texts_by_type[ent_type][i], (x, y), textcoords="offset points", xytext=(5,5), ha='center', fontsize=9)
                
                plt.grid(True)
                plt.show()
            else:
                print(f"Not enough valid '{ent_type}' entity vectors for t-SNE visualization after filtering.")
        else:
            print(f"Not enough '{ent_type}' entity vectors for t-SNE visualization.")


def run_llm_ner(text_data: dict):
    print("\n--- Running LLM Named Entity Recognition ---")
    
    llm_output_directory = "llm_output_jsons/"
    os.makedirs(llm_output_directory, exist_ok=True)

    # Changed to TinyLlama for faster computation
    local_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
    huggingface_token = "" # Your Hugging Face token

    try:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, token=huggingface_token)
        model = AutoModelForCausalLM.from_pretrained(local_model_path, token=huggingface_token)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"Local LLM model loaded from: {local_model_path}")
    except Exception as e:
        print(f"Error loading local LLM model from {local_model_path}: {e}")
        print("Please ensure the model path is correct and 'transformers' and 'torch' (or 'tensorflow') are installed.")
        print("You might need to download the model first, e.g., using Hugging Face CLI or `from_pretrained` with `local_files_only=False` for first-time download.")
        return

    system_prompt = """
    Sie sind ein Experte für die Extraktion strukturierter Informationen aus Texten, speziell für die Katalogisierung von Museumsfotosammlungen.
    Ihre Aufgabe ist es, alle Details für 'Ort', 'Beschreibung', 'Datum', 'Fotograf' und 'Film' aus dem bereitgestellten deutschen Text zu identifizieren und zu extrahieren.
    Geben Sie die extrahierten Informationen als JSON-Objekt zurück. Alle genannten Schlüssel müssen im JSON-Objekt vorhanden sein. Wenn ein Feld nicht explizit im Text gefunden wird, setzen Sie den Wert auf null.
    Seien Sie präzise bei Ihren Extraktionen.
    """

    all_fill_percentages = []
    expected_keys = ["Ort", "Beschreibung", "Datum", "Fotograf", "Film"]

    for doc_name, text in text_data.items():
        print(f"\nProcessing document with LLM: {doc_name}")
        user_prompt = f"Extrahieren Sie die folgenden Entitäten aus dem untenstehenden Text:\n\nText: \"\"\"\n{text}\n\"\"\"\n\nGeben Sie die Ausgabe im JSON-Format mit den Schlüsseln: Ort, Beschreibung, Datum, Fotograf, Film zurück."
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        print("\n--- Local LLM Inference ---")
        print("Prompt:\n", full_prompt)
        print("\nRunning local LLM inference... Please wait.")

        try:
            inputs = tokenizer(full_prompt, return_tensors="pt")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=200,
                num_return_sequences=1,
                do_sample=True,
                top_k=50
            )
            
            llm_response_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            llm_response_text = llm_response_text[len(full_prompt):].strip()

            print("\nLLM Raw Response:\n", llm_response_text)

            try:
                json_start = llm_response_text.find('{')
                json_end = llm_response_text.rfind('}') + 1
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_string = llm_response_text[json_start:json_end]
                    llm_output_json = json.loads(json_string)
                    print("\nLLM Parsed Output (JSON):\n", json.dumps(llm_output_json, indent=2, ensure_ascii=False))

                    output_file_path = os.path.join(llm_output_directory, f"{doc_name}_llm_output.json")
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        json.dump(llm_output_json, f, indent=2, ensure_ascii=False)
                    print(f"LLM output saved to: {output_file_path}")

                else:
                    print("\nCould not find valid JSON in LLM response.")
                    print(f"Full LLM response: {llm_response_text}")

            except json.JSONDecodeError:
                print("\nLLM response was not valid JSON. Please check the LLM's output format or adjust prompt.")
            except Exception as e:
                print(f"\nAn error occurred during LLM output processing: {e}")

        except Exception as e:
            print(f"\nAn error occurred during local LLM inference: {e}")
        
        print("\n--- End Local LLM Inference ---")

# New function to evaluate LLM outputs
def evaluate_llm_outputs(llm_output_directory: str):
    print("\n--- Evaluating LLM Output Metrics ---")
    all_fill_percentages = []
    expected_keys = ["Ort", "Beschreibung", "Datum", "Fotograf", "Film"]

    if not os.path.exists(llm_output_directory):
        print(f"Error: LLM output directory '{llm_output_directory}' not found.")
        return

    json_files = [f for f in os.listdir(llm_output_directory) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in '{llm_output_directory}' to evaluate.")
        return

    for filename in sorted(json_files):
        file_path = os.path.join(llm_output_directory, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                llm_output_json = json.load(f)
            
            doc_name = filename.replace("_llm_output.json", "")
            filled_count = sum(1 for key in expected_keys if llm_output_json.get(key) is not None and llm_output_json.get(key) != "")
            percentage_filled = (filled_count / len(expected_keys)) * 100
            all_fill_percentages.append(percentage_filled)
            print(f"Entities filled for {doc_name}: {filled_count}/{len(expected_keys)} ({percentage_filled:.2f}%)")

        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON from file {filename}. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing file {filename}: {e}")

    if all_fill_percentages:
        average_fill_percentage = np.mean(all_fill_percentages)
        print(f"\n--- LLM Entity Fill Metric Summary ---")
        print(f"Average percentage of entities filled across all documents: {average_fill_percentage:.2f}%")
    else:
        print("\nNo LLM entity fill metrics to report from evaluation.")


def main():
    # Use a local relative path for OCR output directory
    ocr_output_directory = "./cropped_images/txt_color/" 
    llm_output_directory = "llm_output_jsons/" # Define the directory for LLM outputs

    # Ensure the OCR output directory exists locally
    if not os.path.exists(ocr_output_directory):
        print(f"Error: OCR output directory '{ocr_output_directory}' not found.")
        print("Please ensure your OCR results are saved in this location.")
        # You might want to create it or exit, depending on desired behavior
        # os.makedirs(ocr_output_directory, exist_ok=True) 
        return

    text_data = load_text_from_ocr(ocr_output_directory)
    if not text_data:
        print(f"No text files found in '{ocr_output_directory}'. Please check the directory and its contents.")
        return

    run_spacy_ner(text_data)

    run_llm_ner(text_data) 

    evaluate_llm_outputs(llm_output_directory)

    print("\n--- Task C Complete ---")


if __name__ == "__main__":
    main()