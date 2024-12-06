from user_profile import get_user_profile,add_dictionary_to_mongo,vector_db_user_profile
from recommend_meal import get_query_from_lm_studio,optimize_meal_query
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient



if __name__ == "__main__":

    # Initialize the sentence transformer model
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Sample dictionary to insert
    user_profile = get_user_profile()

    # Add dictionary to MongoDB
    result = add_dictionary_to_mongo("Food_recommendation", "user_profile", user_profile)
    
    if result:
        print("Successfull added")
    else:
        print("Operation failed")  # Optional: Handle the failure case

    # Process advanced user query through LM Studio

    # # Taking input from the user
    # user_query = input("Enter your query: ")

    # user_query = "0 saturated fat meals"
    # user_query = "Suggest me full day meals"
    user_query = input("Enter your meal suggestion query: ").strip()
    optimized_user_query = optimize_meal_query(user_query)

    # Filtered dictionary
    user_data = {
        'weight_kg': user_profile['weight_kg'],
        'height_cm': user_profile['height_cm'],
        'dietary_restrictions': user_profile['dietary_restrictions'],
        'dietary_preferences': user_profile['dietary_preferences']
    }

    user_info = optimized_user_query + str(user_data)

    # Use the processed query to search in Qdrant
    query_vector = encoder.encode(user_info).tolist()
    
    client = QdrantClient(url="http://localhost:6338")
    
    hits = client.search(
        collection_name="meals",
        query_vector=query_vector,
        limit=20,
    )

    meals = ""

    # Display the top results
    for hit in hits:
        # print(hit.payload)
        meals += str(hit.payload) + " "

    # print(meals)

    processed_query = get_query_from_lm_studio(user_info,meals)
    print(processed_query)




