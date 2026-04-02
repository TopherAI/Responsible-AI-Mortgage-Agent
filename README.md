# Responsible AI Mortgage Agent

A compliant, auditable multi-agent mortgage pre-approval system built on LangGraph, HubSpot Breeze, and RAG over Fannie Mae, Freddie Mac, FHA, and VA guidelines.

## Architecture
HubSpot Lead → Lead Parser → Compliance RAG → Pre-Approval Agent → Governance/Audit Node

## Stack
- LangGraph (multi-agent graph)
- ChromaDB (compliance vector store)
- HubSpot Breeze (lead ingestion)
- OpenAI Embeddings
- Fannie Mae / Freddie Mac / FHA / VA guidelines

## Structure
- agents/ — LangGraph graph and nodes
- rag/ — guideline ingestion and retriever
- data/guidelines/ — source PDFs
- governance/ — audit and bias check logic
