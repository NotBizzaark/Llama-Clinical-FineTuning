#RAG-Guided LLM Rules
##Goal: 
Use Retrieval-Augmented Generation (RAG) to help a Large Language Model (LLM) consistently follow specific rules, guidelines, or policies.
##🚀 Overview
This project demonstrates how to teach or constrain an LLM by retrieving domain-specific rules and guidelines at runtime.
##Key ideas:
Central rulebase: Store policies, instructions, or best practices in a searchable knowledge base.
Retriever + LLM: At query time, relevant rules are pulled in and provided as context to the model.
Dynamic enforcement: The model’s outputs are steered by the retrieved guidance—no fine-tuning required.
###✨ Features
Auditability: Easily trace which rules influenced each response.
###🛠️ Tech Stack
Vector store: FAISS by default, configurable to others.
