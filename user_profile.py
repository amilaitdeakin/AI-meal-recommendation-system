from pymongo import MongoClient
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from uuid import uuid4  # Import to generate a UUID


def get_user_profile():
  
    # Collect user inputs
    name = input("Enter your name: ").strip()
    
    while True:
        try:
            weight = float(input("Enter your weight in kilograms: ").strip())
            if weight <= 0:
                raise ValueError("Weight must be a positive number.")
            break
        except ValueError as e:
            print(f"Invalid input. Please enter a valid weight. ({e})")
    
    while True:
        try:
            height = float(input("Enter your height in centimeters: ").strip())
            if height <= 0:
                raise ValueError("Height must be a positive number.")
            break
        except ValueError as e:
            print(f"Invalid input. Please enter a valid height. ({e})")
    
    dietary_restrictions = input(
        "Enter any dietary restrictions (e.g., gluten-free, nut allergy). If none, type 'none': "
    ).strip()
    dietary_preferences = input(
        "Enter your dietary preferences (e.g., vegan, keto, high protein). If none, type 'none': "
    ).strip()

    # Store inputs in a dictionary
    user_profile = {
        "name": name,
        "weight_kg": weight,
        "height_cm": height,
        "dietary_restrictions": dietary_restrictions if dietary_restrictions.lower() != "none" else None,
        "dietary_preferences": dietary_preferences if dietary_preferences.lower() != "none" else None,
    }

    return user_profile

def add_dictionary_to_mongo(my_db, my_collection, my_dic_data):

    try:
        # Connect to the MongoDB server (adjust the URI as needed)
        client = MongoClient("mongodb://localhost:27017/")
        
        # Access the database and collection
        db = client[my_db]
        collection = db[my_collection]
        
        # Insert the dictionary as a document
        result = collection.insert_one(my_dic_data)
        
        # Return the inserted ID
        return {"inserted_id": str(result.inserted_id)}
    except Exception as e:
        return {"error": str(e)}
    


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
        dietary_restrictions_vector = encoder.encode(profile["dietary_restrictions"] or "")
        dietary_preferences_vector = encoder.encode(profile["dietary_preferences"] or "")

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

        # Generate a UUID for Qdrant point ID
        qdrant_point_id = str(uuid4())  # Use a UUID as the Qdrant point ID

        # Prepare payload
        payload = {
            "name": profile["name"],
            "weight_kg": profile["weight_kg"],
            "height_cm": profile["height_cm"],
            "dietary_restrictions": profile["dietary_restrictions"],
            "dietary_preferences": profile["dietary_preferences"],
        }

        # Insert the vector and payload into the collection
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=qdrant_point_id,  # Use the generated UUID
                    vector=combined_vector,
                    payload=payload,
                )
            ],
        )

        return f"Vector for profile '{profile['name']}' successfully saved in Qdrant collection '{collection_name}' with ID '{qdrant_point_id}'."

    except Exception as e:
        return f"An error occurred: {e}"
    
# new_user = get_user_profile()
# print(new_user)


# def vector_db_user_profile(profile, qdrant_url="http://localhost:6338", collection_name="user_profile"):
#     """
#     Creates a vector database from the profile dictionary and saves it in Qdrant.

#     Args:
#         profile (dict): Dictionary containing user profile data.
#         qdrant_url (str): URL of the Qdrant instance (default: "http://localhost:6338").
#         collection_name (str): Name of the collection in Qdrant (default: "user_profiles").
    
#     Returns:
#         str: Message indicating the result of the operation.
#     """
#     try:
#         # Initialize Qdrant client
#         client = QdrantClient(url=qdrant_url)

#         # Initialize the SentenceTransformer model
#         encoder = SentenceTransformer("all-MiniLM-L6-v2")

#         # Extract and encode the textual fields
#         dietary_restrictions_vector = encoder.encode(profile["dietary_restrictions"])
#         dietary_preferences_vector = encoder.encode(profile["dietary_preferences"])

#         # Normalize the numerical fields (scale to [0, 1])
#         weight_normalized = profile["weight_kg"] / 200.0  # Assuming a max weight of 200 kg
#         height_normalized = profile["height_cm"] / 250.0  # Assuming a max height of 250 cm

#         # Combine all features into a single vector
#         combined_vector = list(dietary_restrictions_vector) + list(dietary_preferences_vector) + [
#             weight_normalized,
#             height_normalized,
#         ]

#         # Create the Qdrant collection if it doesn't exist
#         client.recreate_collection(
#             collection_name=collection_name,
#             vectors_config=models.VectorParams(
#                 size=len(combined_vector),  # Length of the combined vector
#                 distance=models.Distance.COSINE,
#             ),
#         )

#         # Insert the vector and payload into the collection
#         payload = {
#             "name": profile["name"],
#             "weight_kg": profile["weight_kg"],
#             "height_cm": profile["height_cm"],
#             "dietary_restrictions": profile["dietary_restrictions"],
#             "dietary_preferences": profile["dietary_preferences"],
#             "user_id": str(profile["_id"]),  # Convert ObjectId to string for JSON compatibility
#         }

#         client.upsert(
#             collection_name=collection_name,
#             points=[
#                 models.PointStruct(
#                     id=str(profile["_id"]),  # Use the MongoDB ObjectId as the ID
#                     vector=combined_vector,
#                     payload=payload,
#                 )
#             ],
#         )

#         return f"Vector for profile '{profile['name']}' successfully saved in Qdrant collection '{collection_name}'."

#     except Exception as e:
#         return f"An error occurred: {e}"

    


