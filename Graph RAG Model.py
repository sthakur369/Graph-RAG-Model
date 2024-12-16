# **Imports**

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DistilBertTokenizer
import re
import nltk
from nltk.tokenize import sent_tokenize
from pyvis.network import Network
import networkx as nx
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import spacy
import wikipedia
import torch
import warnings
warnings.filterwarnings("ignore")

# Download the necessary NLTK resources (run this only once)
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

# **Text to Entity Extraction**

# **Load the Input text**
page = wikipedia.page('The French Revolution')
full_text = page.content

# Print the full text
print(full_text)

# **Set your Questions**
queries = [
    "When did the French Revolution start?",
    "Who led the coup of 18 Brumaire in November 1799?",
    "What was the name of the assembly that replaced the monarchy in September 1792?",
    "What was the name of the radical club that played a significant role in the Revolution?",
    "What was the main cause of French Revolution?"
]

# **Load the model**
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
ent_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
ent_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

# Move model to GPU if available
ent_model.to(device)

# Create the pipeline with the model and tokenizer
triplet_extractor = pipeline('text2text-generation', model=ent_model,
                             tokenizer=ent_tokenizer, device=0 if torch.cuda.is_available() else -1)

# **Split the Text**


def clean_text(text):
    # Remove citation markers like [6], [7], etc.
    text = re.sub(r'$$\d+$$', '', text)

    # Remove section headers like "== Causes =="
    text = re.sub(r'==\s*[^=]+\s*==', '', text)

    # Remove special characters except full stops
    text = re.sub(r'[^\w\s.]', '', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def split_text_into_sentences(text):
    return sent_tokenize(text)


def split_text_into_batches(sentences, batch_size):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i:i + batch_size]


def split_text_into_chunks(cleaned_text, batch_size=10):
    # Split the full text into sentences
    sentences = split_text_into_sentences(cleaned_text)

    # Specify batch size for processing sentences
    batch_size = 10

    # Store all processed sentences in a list
    all_sentences = []

    # Iterate over batches of sentences
    for batch in split_text_into_batches(sentences, batch_size):
        for sentence in batch:
            all_sentences.append(sentence)  # Store each sentence in the list

    return all_sentences


# Split the text
cleaned_text = clean_text(full_text)
split_text = split_text_into_chunks(cleaned_text)

print(len(split_text))

# Print each chunk
for i, chunk in enumerate(split_text, 1):
    print(f"Chunk {i}:\n{chunk}\n")

# **Extract the Entities**
# Function to parse the generated text and extract the triplets


def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append(
                    {'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append(
                    {'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append(
            {'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
    return triplets


# Process each chunk
extracted_triplets = []
for idx, chunk in enumerate(split_text, 1):
    extracted_text = triplet_extractor.tokenizer.batch_decode(
        [triplet_extractor(chunk, return_tensors=True, return_text=False)[
            0]["generated_token_ids"]]
    )
    extracted_triplets_chunk = extract_triplets(extracted_text[0])
    extracted_triplets.extend(extracted_triplets_chunk)

print(len(extracted_triplets))
print(extracted_triplets)

# **Graph Creation**
data = extracted_triplets.copy()

# **Create the Graph**
# NetworkX graph
G = nx.DiGraph()

# Add nodes and edges to the graph
for entry in data:
    head = entry['head']
    tail = entry['tail']
    relation = entry['type']

    G.add_node(head)
    G.add_node(tail)

    G.add_edge(head, tail, relation=relation)

G

# **Visulize the Graph**
# Pyvis Network for visulization
net = Network(notebook=True, filter_menu=True, select_menu=True)

head_color = '#81C784'   # Light Green
tail_color = '#FFB74D'   # Light Salmon
edge_color = '#1E88E5'    # Blue
font_color = '#212121'     # Black

# Add nodes and edges to Pyvis network
for node in G.nodes():
    color = head_color if any(G.successors(node)) else tail_color
    net.add_node(node, label=node, color=color, title=node)

for edge in G.edges(data=True):
    head, tail, attrs = edge
    relation = attrs['relation']
    net.add_edge(
        head,
        tail,
        title=relation,
        label=relation,
        arrows='to',
        color=edge_color,
        dashes=False
    )

net.force_atlas_2based()

# Show the graph directly in the notebook
net.show("Graph.html")

# **RAG Model**
device_id = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **Loading multiple models to get response**
qa_pipelines = {
    'transformer_default': pipeline("question-answering", device=device_id, max_answer_len=2048)
}

# **Extract Entities from the Question**


def extract_entities_from_query(query):
    doc = nlp(query)

    # Extract named entities
    entities = [ent.text for ent in doc.ents]

    # Extract nouns
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']

    # Combine entities and nouns
    all_entities_and_nouns = entities + nouns

    return all_entities_and_nouns

# **Extract subgraph**


def tokenize_and_stem(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(
        word) for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens


def fuzzy_match(entity_tokens, node_tokens):
    entity_set = set(entity_tokens)
    node_set = set(node_tokens)
    intersection = entity_set.intersection(node_set)
    return len(intersection) / max(len(entity_set), 1)


def retrieve_subgraph(query, graph, max_depth=4, match_threshold=0.4):
    """Retrieve a subgraph relevant to the query."""
    entities = extract_entities_from_query(query)
    subgraph = nx.MultiDiGraph()

    for entity in entities:
        entity_tokens = tokenize_and_stem(entity)
        for node in graph.nodes:
            node_tokens = tokenize_and_stem(node)
            match_score = fuzzy_match(entity_tokens, node_tokens)
            if match_score >= match_threshold:
                for _, neighbor in nx.bfs_edges(graph, source=node, depth_limit=max_depth):
                    try:
                        relation = graph[node][neighbor][0].get(
                            'relation', 'related_to')
                        subgraph.add_edge(node, neighbor, relation=relation)
                    except (KeyError, IndexError):
                        subgraph.add_edge(
                            node, neighbor, relation='related_to')

    return subgraph

# **Subgraph in Textual format**


def convert_subgraph_to_text(subgraph):
    triples = []
    for u, v, data in subgraph.edges(data=True):
        relation = data.get('relation', 'related_to')
        triples.append({'head': u, 'type': relation, 'tail': v})

    # Convert the list of triples to a formatted string
    formatted_triples = [
        f"{{'head': '{triple['head']}', 'type': '{triple['type']}', 'tail': '{triple['tail']}'}}" for triple in triples]
    return "[\n" + ",\n".join(formatted_triples) + "\n]"

# **Generate Response**


def generate_response(query, graph):
    subgraph = retrieve_subgraph(query, graph)

    if not subgraph.nodes:
        combined_list = []
        response = {
            'subgraph':  nx.MultiDiGraph(),
            'subgraph_text': None,
            'responses':
                {'None':
                 {'answer': 'Sorry, I could not find any relevant information.'
                  }
                 }
        }

        return combined_list, response

    subgraph_text = convert_subgraph_to_text(subgraph)

    responses = {name: pipeline({'question': query, 'context': subgraph_text})
                 for name, pipeline in qa_pipelines.items()}

    combined_list = list(responses.values())

    response = {
        "subgraph": subgraph,
        "subgraph_text": subgraph_text,
        "responses": responses
    }

    return combined_list, response

# **Visulize subgraph**


def visualize_subgraph(subgraph, query):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(subgraph)
    nx.draw(subgraph, pos, with_labels=True, node_size=3000, node_color="skyblue",
            font_size=10, font_color="black", font_weight="bold", edge_color="gray")
    edge_labels = {(u, v): data.get('relation', 'related_to')
                   for u, v, data in subgraph.edges(data=True)}
    nx.draw_networkx_edge_labels(
        subgraph, pos, edge_labels=edge_labels, font_color="red")
    plt.title(f"Subgraph for query: {query}")
    plt.show()


# **Enhance the Response**
# Initialize the model
enhance_pipe = pipeline(
    "text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", max_length=1024, device=device_id)


def enhance_answer(query, answer):

    sys_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    user_prompt = f"""Given the user input: '{{ "Query": "{query}", "Answer": "{answer}" }}', enhance the answer based on the provided Query and Answer. Provide a direct and descriptive response without mentioning that you are enhancing the answer. Avoid any unnecessary explanations or notes at the end."""

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    enhanced_response = enhance_pipe(messages)
    return enhanced_response[0]['generated_text'][-1]['content']

# **Run the RAG Model**

# **Simple Run: Without showing subgraphs. Only showing Answers**


def main():
    print('Main Graph details:', G)
    for query in queries:
        combined_list, response = generate_response(query, G)
        subgraph = response.get('subgraph', nx.MultiDiGraph())
        subgraph_text = response.get('subgraph_text', '')

        # Print the query and subgraph details
        print(f"\n+++++++++++++++ Query: {query} ++++++++")
    #     print("Subgraph Details:")
    #     print(subgraph)
    #     print(subgraph_text)
        print()

        # Handle the case where subgraph_text is None
        if subgraph_text is None:
            print(f"*** Answer: {response['responses']['None']['answer']} ***")
            print()
            continue

        # Print responses from each pipeline
        if 'responses' in response:
            for name, answer in response['responses'].items():
                clean_ans = clean_text(answer['answer'])
                print(f"-> Answer: {clean_ans}")

                # Enhance the answer
                enhanced_answer = enhance_answer(query, clean_ans)
                print(f"-> Enhanced Answer: {enhanced_answer}")
                print()
        else:
            print("*** No responses available. ***")
            print()

        print()

    # **Detailed Run: With showing Answers and Subgraphs**
    print('Main Graph details:', G)
    for query in queries:
        combined_list, response = generate_response(query, G)
        subgraph = response.get('subgraph', nx.MultiDiGraph())
        subgraph_text = response.get('subgraph_text', '')

        # Print the query and subgraph details
        print(f"\n+++++++++++++++ Query: {query} ++++++++++++++++++++++")
        print()

        # Handle the case where subgraph_text is None
        if subgraph_text is None:
            print(f"*** Answer: {response['responses']['None']['answer']} ***")
            print()
            continue

        # Print responses from each pipeline
        if 'responses' in response:
            for name, answer in response['responses'].items():
                clean_ans = clean_text(answer['answer'])
                print(f"-> Answer: {clean_ans}")

                # Enhance the answer
                enhanced_answer = enhance_answer(query, clean_ans)
                print(f"-> Enhanced Answer: {enhanced_answer}")
                print()
        else:
            print("*** No responses available. ***")
            print()

        # Visualize the subgraph
        print('****Subgraph Visualization ***')
        visualize_subgraph(subgraph, query)
        print()
        print("*** Subgraph Details: ***")
        print(subgraph)
        print(subgraph_text)


if __name__ == '__main__':
    main()
