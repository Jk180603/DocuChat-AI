from typing import List, Dict
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.schema import Document
from config.config import settings

class RAGChain:
    """Retrieval-Augmented Generation chain with real LLM"""
    
    def __init__(self, vector_store_manager):
        self.vector_store_manager = vector_store_manager
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Use Ollama with Llama 3.2 (FREE, runs locally!)
        self.llm = Ollama(
            model="llama3.2",
            temperature=0.7,
            num_ctx=4096,  # Context window
        )
        
        self.qa_chain = None
        self.create_qa_chain()
    
    def create_qa_chain(self):
        """Create conversational retrieval chain with proper prompting"""
        
        # Enhanced prompt template
        prompt_template = """You are an intelligent AI assistant helping users understand their documents. 

Use the following pieces of context from the documents to answer the question. If you cannot find the answer in the context, say "I cannot find this information in the provided documents."

IMPORTANT:
- Answer based ONLY on the context provided
- Be specific and cite which document/page when possible
- If asked about dates, numbers, or specific facts, extract them exactly from the context
- If the context doesn't contain the answer, admit it clearly

Context from documents:
{context}

Question: {question}

Detailed Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = self.vector_store_manager.get_retriever(k=settings.RETRIEVAL_K)
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=False,
            chain_type="stuff"  # Use "stuff" method for better answers
        )
        
        return self.qa_chain
    
    def ask_question(self, question: str) -> Dict:
        """Ask a question and get answer with sources"""
        if self.qa_chain is None:
            self.create_qa_chain()
        
        try:
            result = self.qa_chain({"question": question})
            
            # Format sources
            sources = []
            for doc in result.get("source_documents", []):
                sources.append({
                    "content": doc.page_content[:300] + "...",  # Longer preview
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A")
                })
            
            return {
                "answer": result["answer"],
                "sources": sources
            }
        except Exception as e:
            print(f"Error in ask_question: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": []
            }