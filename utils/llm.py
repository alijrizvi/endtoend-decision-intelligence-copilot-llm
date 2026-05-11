


# This upcoming Code will be Archived as Streamlit Cloud cannot install local-only packages like Ollama. The code is preserved here for reference and potential future use in environments that support Ollama.

# import ollama

# from utils.business_context import business_context


# def ask_llm(user_question,
#             revenue,
#             profit,
#             retention):

#     full_prompt = f"""
    
#     LIVE BUSINESS METRICS - Dynamic LLM Integration

#     {business_context}

#     Revenue: {revenue}
#     Profit: {profit}
#     Retention: {retention}

#     USER QUESTION:
#     {user_question}

#     """

#     response = ollama.chat(
#         model = 'llama3',
#         messages = [
#             {
#                 'role': 'user',
#                 'content': full_prompt
#             }
#         ]
#     )

#     return response['message']['content']