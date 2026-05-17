from google import genai
from groq import Groq


def get_gemini_client(gemini_api_key: str) -> genai.Client:
    gemini_client = genai.Client(api_key=gemini_api_key)
    print("Gemini initialized")

    return gemini_client


def get_groq_client(groq_api_key: str) -> Groq:
    groq_client = Groq(api_key=groq_api_key)
    print("Groq initialized")

    return groq_client
