from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import json


# Initialize the sentence transformer model
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Load the JSON data
with open("Food_recommendation.meal_collection.json", "r") as file:
    data = json.load(file)

# Prepare documents and payloads in one step
documents, payloads = [], []
for restaurant in data:
    restaurant_name = restaurant['restaurant_name']
    for product in restaurant['products']:
        # Create a descriptive string for each product
        description = (
            f"{product['name']} with {product['calories']} calories, "
            f"{product['total fat']} fat, {product['saturated fat']} saturated fat, "
            f"{product['cholesterol']} cholesterol, {product['total carb']} carbs, "
            f"{product['dietary fibre']} dietary fibre, {product['sugar']} sugar, "
            f"{product['protein']} protein"
        )
        documents.append(description)
        payloads.append({
            "name": product["name"],
            "calories": product["calories"],
            "total_fat": product["total fat"],
            "saturated_fat": product["saturated fat"],
            "cholesterol": product["cholesterol"],
            "total_carb": product["total carb"],
            "dietary_fibre": product["dietary fibre"],
            "sugar": product["sugar"],
            "protein": product["protein"]
        })

# Batch encode all documents
document_vectors = encoder.encode(documents, convert_to_tensor=False)

# Initialize the Qdrant client (use a persistent or properly configured in-memory storage)
client = QdrantClient(url="http://localhost:6338")


# Create a collection in Qdrant
client.create_collection(
    collection_name="meals",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size defined by model
        distance=models.Distance.COSINE,
    ),
)

# Upload points to the Qdrant collection
points = [
    models.PointStruct(
        id=idx,
        vector=vector.tolist(),
        payload=payload
    )
    for idx, (vector, payload) in enumerate(zip(document_vectors, payloads))
]
client.upload_points(collection_name="meals", points=points)