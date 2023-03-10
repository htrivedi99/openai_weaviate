import openai
from openai.embeddings_utils import get_embedding
import pandas as pd
import weaviate

openai.api_key = "YOUR-OPENAI-API-KEY"
client = weaviate.Client(url="http://localhost:8080")


def read_json_file():
    filename = "history_text.json"
    df = pd.read_json(filename)
    return df


def generate_data_embeddings(df):
    df['embedding'] = df['text'].apply(lambda row: get_embedding(row, engine="text-embedding-ada-002"))
    return df


def weaviate_create_schema():
    schema = {
        "classes": [{
            "class": "HistoryText",
            "description": "Contains the paragraphs of text along with their embeddings",
            "vectorizer": "none",
            "properties": [{
                "name": "content",
                "dataType": ["text"],
            }]
        }]
    }
    client.schema.create(schema)


def weaviate_delete_schema():
    client.schema.delete_class("HistoryText")


def weaviate_add_data(df):
    client.batch.configure(batch_size=10)
    with client.batch as batch:
        for index, row in df.iterrows():
            text = row['text']
            ebd = row['embedding']
            batch_data = {
                "content": text
            }
            batch.add_data_object(data_object=batch_data, class_name="HistoryText", vector=ebd)

    print("Data Added!")


def query(input_text, k):
    input_embedding = get_embedding(input_text, engine="text-embedding-ada-002")
    vec = {"vector": input_embedding}
    result = client \
        .query.get("HistoryText", ["content", "_additional {certainty}"]) \
        .with_near_vector(vec) \
        .with_limit(k) \
        .do()

    output = []
    closest_paragraphs = result.get('data').get('Get').get('HistoryText')
    for p in closest_paragraphs:
        output.append(p.get('content'))

    return output


if __name__ == "__main__":
    dataframe = read_json_file()
    dataframe = generate_data_embeddings(dataframe)
    weaviate_create_schema()
    weaviate_add_data(dataframe)
    # Run the above 4 lines only once!
    
    input_text = "american economy"
    k_vectors = 3

    result = query(input_text, k_vectors)
    for text in result:
        print(text)
