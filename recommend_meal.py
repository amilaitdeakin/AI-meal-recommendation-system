from qdrant_client import QdrantClient

import requests
from nltk.corpus import stopwords
from user_profile import get_user_profile



# Define a function to clean the query
def optimize_meal_query(user_query):

    # Define relevant keywords
    remove_keywords = {"can", "you", "suggest", "I", "need", "what", "and", "a","the","I am",}
    
    # Tokenize the query into words
    words = user_query.lower().split()
    
    # Get stop words
    stop_words = set(stopwords.words("english"))
    
    # Filter words: Keep only relevant and non-stop words
    optimized_words = [
        word for word in words if word not in remove_keywords and word not in stop_words
    ]
    
    # Join the words back into a cleaned query
    optimized_query = " ".join(optimized_words)
    return optimized_query


def get_query_from_lm_studio(query,knowledge_db):

    # Initialize the Qdrant client
    client = QdrantClient(url="http://localhost:6338")

    lm_studio_url = "http://127.0.0.1:1234/v1/chat/completions"  # Replace with your LM Studio endpoint
    payload = {
        "model": "llama-3.2-1b-instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You must answer strictly only the information provided in the meals knowledge database "
                    "and from the user profile details in the query. Do not use external knowledge, make assumptions, "
                    "or generate results outside the database. If the information is not available, respond with: "
                    "'Information not available in the database.'\n\n"
                    f"Meals knowledge database: {knowledge_db}\n"
                    f"User details: {query}"
                )
            },
            {"role": "user", "content": query}
        ],
        "temperature": 0.5,
        "max_tokens":200,  # Adjust this based on your needs
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(lm_studio_url, json=payload, headers=headers)

    try:
        response = requests.post(lm_studio_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise HTTP errors if any
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error from LM Studio: {e}")


