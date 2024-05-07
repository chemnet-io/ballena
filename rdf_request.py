import os
import rdflib
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

os.environ['OPENAI_API_KEY'] = ''
OPENAI_API_KEY = ''

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

entities = []

g = rdflib.Graph()

def fetch_sparql_results():
    sparql_result = []
    offset = 0
    sparql_batch_size = 100  # Batch size for SPARQL queries
    more_results = True

    while more_results:
        try:
            sparql_query = f"""
                            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                            select ?label where {{
                                SERVICE <https://nubbekg.aksw.org/sparql> {{
                                    ?s a <http://nubbekg.aksw.org/ontology#IsolationSite> .
                                    ?s rdfs:label ?label .
                                }}
                            }} LIMIT {sparql_batch_size} OFFSET {offset}
                            """
            current_results = g.query(sparql_query)
            current_results = list(current_results)
            if not current_results or len(current_results) < sparql_batch_size:
                more_results = False
            else:
                offset += sparql_batch_size
                sparql_result.extend(current_results)
        except Exception as e:
            print("Exception when querying SPARQL endpoint" + ": %s" % e)
            break

    return sparql_result

sparq_result = fetch_sparql_results()

for entry in sparq_result:
    label = entry.label
    doc = Document(page_content=label, metadata={'label': label})
    entities.append(doc)

# Create FAISS index from documents
faiss_index = FAISS.from_documents(entities, embeddings)

# Save the FAISS index locally
faiss_index.save_local("faiss_index")

# Load the FAISS index, allowing deserialization
loaded_faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Perform a similarity search
query = "Minas Gerais"
docs_with_score = loaded_faiss_index.similarity_search_with_score(query, top_k=5)

for doc, score in docs_with_score:
    print(f"Document: {doc.page_content}, Score: {score}")