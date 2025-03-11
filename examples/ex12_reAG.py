# https://medium.com/nerd-for-tech/fixing-rag-with-reasoning-augmented-generation-919939045789

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import os
os.environ["GROQ_API_KEY"] = "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"
#
llm_relevancy = ChatGroq(
     model="llama-3.3-70b-versatile",
    temperature=0,)

#

llm = ChatOllama(model="deepseek-r1:14b",
                 temperature=0.6,
                 max_tokens=3000,
                )

REAG_SYSTEM_PROMPT = """
# Role and Objective
You are an intelligent knowledge retrieval assistant. Your task is to analyze provided documents or URLs to extract the most relevant information for user queries.

# Instructions
1. Analyze the user's query carefully to identify key concepts and requirements.
2. Search through the provided sources for relevant information and output the relevant parts in the 'content' field.
3. If you cannot find the necessary information in the documents, return 'isIrrelevant: true', otherwise return 'isIrrelevant: false'.

# Constraints
- Do not make assumptions beyond available data
- Clearly indicate if relevant information is not found
- Maintain objectivity in source selection
"""

rag_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

from pydantic import BaseModel,Field
from typing import List
from langchain_core.output_parsers import JsonOutputParser

class ResponseSchema(BaseModel):
    content: str = Field(...,description="The page content of the document that is relevant or sufficient to answer the question asked")
    reasoning: str = Field(...,description="The reasoning for selecting The page content with respect to the question asked")
    is_irrelevant: bool = Field(...,description="Specify 'True' if the content in the document is not sufficient or relevant to answer the question asked otherwise specify 'False' if the context or page content is relevant to answer the question asked")


class RelevancySchemaMessage(BaseModel):
    source: ResponseSchema

relevancy_parser = JsonOutputParser(pydantic_object=RelevancySchemaMessage)

from langchain_community.document_loaders import PyMuPDFLoader

file_path = "./data/fibromyalgia.pdf"
loader = PyMuPDFLoader(file_path)
#
docs = loader.load()
print(len(docs))
print(docs[0].metadata)

from langchain.schema import Document
def format_doc(doc: Document) -> str:
    return f"Document_Title: {doc.metadata['title']}\nPage: {doc.metadata['page']}\nContent: {doc.page_content}"

### Helper function to extract relevant context
from langchain_core.prompts import PromptTemplate
def extract_relevant_context(question,documents):
    result = []
    for doc in documents:
        formatted_documents = format_doc(doc)
        system = f"{REAG_SYSTEM_PROMPT}\n\n# Available source\n\n{formatted_documents}"
        prompt = f"""Determine if the 'Avaiable source' content supplied is sufficient and relevant to ANSWER the QUESTION asked.
        QUESTION: {question}
        #INSTRUCTIONS TO FOLLOW
        1. Analyze the context provided thoroughly to check its relevancy to help formulizing a response for the QUESTION asked.
        2, STRICTLY PROVIDE THE RESPONSE IN A JSON STRUCTURE AS DESCRIBED BELOW:
            ```json
               {{"content":<<The page content of the document that is relevant or sufficient to answer the question asked>>,
                 "reasoning":<<The reasoning for selecting The page content with respect to the question asked>>,
                 "is_irrelevant":<<Specify 'True' if the content in the document is not sufficient or relevant.Specify 'False' if the page content is sufficient to answer the QUESTION>>
                 }}
            ```
         """
        messages =[ {"role": "system", "content": system},
                       {"role": "user", "content": prompt},
                    ]
        response = llm_relevancy.invoke(messages)    
        print(response.content)
        formatted_response = relevancy_parser.parse(response.content)
        result.append(formatted_response)
    final_context = []
    for items in result:
        if (items['is_irrelevant'] == False) or ( items['is_irrelevant'] == 'false') or (items['is_irrelevant'] == 'False'):
            final_context.append(items['content'])
    return final_context

question = "What is Fibromyalgia?"
final_context = extract_relevant_context(question,docs)
print(len(final_context))

def generate_response(question,final_context):
    prompt = PromptTemplate(template=rag_prompt,
                                     input_variables=["question","context"],)
    chain  = prompt | llm_relevancy
    response = chain.invoke({"question":question,"context":final_context})
    # print(response.content.split("\n\n")[-1])
    return response.content.split("\n\n")[-1]

final_response = generate_response(question,final_context)
print(final_response)