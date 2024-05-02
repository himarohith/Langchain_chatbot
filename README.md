BUSINESS PROBLEM STATEMENT: 

Streamlining MS BAIS student inquiries is vital. Leveraging OpenAI's LangChain, our
project creates a specialized chatbot using CHAT GPT. This intelligent solution
addresses admission and general queries, easing the Graduate Coordinator's
workload. By maximizing LangChain's capabilities, we aim to enhance efficiency and
elevate the student and staff experience.

TECHNICAL ARCHITECTURE:
1) Response Generation and User Interaction:
2) Prompt Development: Formulate prompts for user
queries and connect them to OpenAI's Lang Chain
model GPT-3.5 Turbo using the OpenAI API.
3) Document Retrieval: When users pose questions, fetch
relevant documents from the Pinecone vector database
for analysis.
4) Summarization and Response: Utilize the Lang Chain
model for summarization, generating responses based
on the highest-ranked information. Display the
summarized response to the user through the interface.

TOOLS AND TECHNOLOGIES: 
• Pinecone vector database
• Open ai embedding model text-embedding-ada-02
• Lang chain model gpt-3.5 turbo-16k.
• Pinecone API
• Open ai API
• Streamlit
• Langchain library
• Sqlite
• SMTP (simple mail transfer protocol) client.
• Multipurpose Internet Mail Extensions (MIME)

FUTURE SCOPE :
1) Currently storing the entire metadata as well as the
conversation in the database. However, we want to filter
the conversation of the user and display it.
2) Implement the security feature of the bot by generating
the security code sending it to email and validating it.
3) If the bot is unable to answer USF-related questions, then
generate a Jira ticket based on the UID number and send
it to the corresponding department handling person.
Storing the user feedback into the database and want to
use this information for future enhancements,
implementations, and metrics of the bot.
