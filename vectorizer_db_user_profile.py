from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from bson.objectid import ObjectId

def vector_db_user_profile(profile, qdrant_url="http://localhost:6338", collection_name="user_profile"):
    """
    Creates a vector database from the profile dictionary and saves it in Qdrant.

    Args:
        profile (dict): Dictionary containing user profile data.
        qdrant_url (str): URL of the Qdrant instance (default: "http://localhost:6338").
        collection_name (str): Name of the collection in Qdrant (default: "user_profiles").
    
    Returns:
        str: Message indicating the result of the operation.
    """
    try:
        # Initialize Qdrant client
        client = QdrantClient(url=qdrant_url)

        # Initialize the SentenceTransformer model
        encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Extract and encode the textual fields
        dietary_restrictions_vector = encoder.encode(profile["dietary_restrictions"])
        dietary_preferences_vector = encoder.encode(profile["dietary_preferences"])

        # Normalize the numerical fields (scale to [0, 1])
        weight_normalized = profile["weight_kg"] / 200.0  # Assuming a max weight of 200 kg
        height_normalized = profile["height_cm"] / 250.0  # Assuming a max height of 250 cm

        # Combine all features into a single vector
        combined_vector = list(dietary_restrictions_vector) + list(dietary_preferences_vector) + [
            weight_normalized,
            height_normalized,
        ]

        # Create the Qdrant collection if it doesn't exist
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(combined_vector),  # Length of the combined vector
                distance=models.Distance.COSINE,
            ),
        )

        # Insert the vector and payload into the collection
        payload = {
            "name": profile["name"],
            "weight_kg": profile["weight_kg"],
            "height_cm": profile["height_cm"],
            "dietary_restrictions": profile["dietary_restrictions"],
            "dietary_preferences": profile["dietary_preferences"],
            "user_id": str(profile["_id"]),  # Convert ObjectId to string for JSON compatibility
        }

        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(profile["_id"]),  # Use the MongoDB ObjectId as the ID
                    vector=combined_vector,
                    payload=payload,
                )
            ],
        )

        return f"Vector for profile '{profile['name']}' successfully saved in Qdrant collection '{collection_name}'."

    except Exception as e:
        return f"An error occurred: {e}"


    # result = create_and_save_vector_db(profile)
    # print(result)



