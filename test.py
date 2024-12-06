from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import json

# Initialize the sentence transformer model
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Load the JSON data
with open("Food_recommendation.meal_collection.json", "r") as file:
    data = json.load(file)

# Prepare documents and payloads
documents = []
payloads = []
for restaurant in data:
    restaurant_name = restaurant['restaurant_name']
    for product in restaurant['products']:
        # Create a descriptive string for each product
        description = (
            f"{product['name']} with {product['calories']} calories, "
            f"{product['total fat']} fat, {product['saturated fat']} saturated fat, " 
            f"{product['cholesterol']} cholesterol, {product['total carb']} carbs, "
            f"{product['dietary fibre']} dietary fibre, {product['sugar']} sugar, "
            f"{product['protein']} protein "
        )
        documents.append(description)
        payloads.append({
            "restaurant_name": restaurant_name,
            "name": product["name"],
            "calories": product["calories"],
            "total_fat": product["total fat"],
            "saturated_fat":product["saturated fat"],
            "cholesterol":product["cholesterol"],
            "total_carb":product["total carb"],
            "dietary_fibre":product["dietary fibre"],
            "sugar":product["sugar"],
            "protein":product["protein"]
        })

# Initialize the Qdrant client
client = QdrantClient(":memory:")

# Create a collection in Qdrant
client.create_collection(
    collection_name="meals",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size defined by model
        distance=models.Distance.COSINE,
    ),
)

# Upload points to the Qdrant collection
client.upload_points(
    collection_name="meals",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc).tolist(),
            payload=payload
        )
        for idx, (doc, payload) in enumerate(zip(documents, payloads))
    ],
)

# Query the Qdrant collection
query_vector = encoder.encode("Suggest a 3 meal plan for 7 days. Weight is 85 Kg, Height is 160 cm. I have cholesterol as well.").tolist()
hits = client.search(
    collection_name="meals",
    query_vector=query_vector,
    limit=5,
)

# Display the top results
for hit in hits:
    print(hit.payload, "score:", hit.score)
