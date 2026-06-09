from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    task='text-generation',
)

global_model = ChatHuggingFace(llm = llm)

embeddings = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

class RAGengine:
    def __init__(self,pdf_path,model=None):
        self.loader = PyPDFLoader(pdf_path)
        documents = self.loader.lazy_load()

        self.docs=[]

        for doc in documents:
            self.docs.append(doc)

        self.embeddings = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
        self.model = global_model if model is None else model

        text_splitter = SemanticChunker(
            self.embeddings,breakpoint_threshold_type='percentile'
        )

        self.chunks = text_splitter.split_documents(self.docs)
        
        self.parser = StrOutputParser()

        self.vector_store = FAISS.from_documents(
            embedding=self.embeddings,
            documents=self.chunks
        )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":10}
        )

        self.bm25_retriever = BM25Retriever.from_documents(documents=self.chunks)
        self.bm25_retriever.k=10

        self.compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12")


    def generate_hypothetical_answer(self,query):
        self.hypothetical_template = """Rewrite the user's question into a detailed search query
                that would retrieve the most relevant document passages.

                Question:
                {query}"""
        self.prompt = PromptTemplate.from_template(self.hypothetical_template)
        self.chain = self.prompt | self.model | self.parser
        return self.chain.invoke({"query":query})
    
    def advance_retrieve(self,query):
        hypo_ans = self.generate_hypothetical_answer(query)

        vector_docs = self.retriever.invoke(hypo_ans)
        bm25_docs = self.bm25_retriever.invoke(hypo_ans)

        combined_res = (vector_docs+bm25_docs)
        unique_docs= {}
        for doc in combined_res:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content]=doc

        final_combined_docs = list(unique_docs.values())

        final_docs = self.compressor.compress_documents(final_combined_docs,query)
        return final_docs[:5]
    
    def generate_response(self,query,history=None):
        best_chunks = self.advance_retrieve(query)

        if not best_chunks:
            return "I couldn't find relevant information in the uploaded document."
        
        chat_history = ""
        if history:
            chat_history="\n".join(
                [
                    f"{msg['role']} : {msg['content']}" for msg in history
                ]
            )

        template = PromptTemplate(
            template="""You are a research assistant.

                        Use ONLY the provided context.

                        Previous conversation is provided only to resolve references
                        such as "he", "that company", or "the previous topic".

                        All factual answers must come from the provided context.

                        If the answer cannot be found in the context,
                        respond exactly with:

                        "I don't know based on the provided document."

                        Provide a detailed explanation.
                        When possible:
                        - Explain the concept in 2-4 paragraphs.
                        - Include important features, components, objectives, and applications.
                        - Summarize key points from the context.
                        - Do not make up information.

                        Previous conversation:
                        {history}

                        Context:
                        {context}

                        Question:
                        {question}
                        """,
            input_variables=['history','context','question']
        )

        final_context = "\n\n".join([doc.page_content for doc in best_chunks])

        final_chain = template | self.model | self.parser

        results = final_chain.invoke(({'history':chat_history,'context':final_context,'question':query}))

        #adding the pages from where the data was obtained
        sources=[]

        for doc in best_chunks:
            page = doc.metadata.get("page")

            if page is not None:
                sources.append(f"page {page+1}")

        sources = sorted(set(sources))

        if sources:
            results += "\n\n---\n**Sources:** " + ", ".join(sources)


        return results