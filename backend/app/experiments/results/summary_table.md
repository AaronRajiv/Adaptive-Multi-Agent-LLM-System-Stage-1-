# Comparative Multi-Agent Orchestration Benchmark Report

| System | Tasks | Success % | Quality (0-100) | Citation Valid % | Avg Latency (ms) | Avg Tokens | Total Cost ($) | Failure % | Avg Replans |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct LLM** | 32 | 100.0% | 88.0 | 100.0% | 0.0 | 56 | $0.0003 | 0.0% | 0.00 |
| **Naive RAG** | 32 | 100.0% | 88.0 | 100.0% | 0.1 | 157 | $0.0005 | 0.0% | 0.00 |
| **Fixed Multi-Agent** | 32 | 100.0% | 88.0 | 93.75% | 1.8 | 10318 | $0.0342 | 0.0% | 0.12 |
| **Adaptive System** | 32 | 100.0% | 88.06 | 93.75% | 1.5 | 9453 | $0.0314 | 0.0% | 0.12 |


### Human-Scored Representative Sample Evaluation (5 Tasks)

| Task ID | Task Category | Prompt | Direct LLM (0-100) | Naive RAG (0-100) | Fixed Multi-Agent (0-100) | Adaptive System (0-100) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **SF-01** | Conversational Greeting | "Hello there! How are you doing today?" | 95 | 90 | 50 | **98** |
| **SDL-01** | Single-Doc Lookup | "According to the 2025 FIA technical regulations, what is the minimum mass of the car without fuel?" | 60 | 92 | 88 | **95** |
| **MDA-01** | Multi-Domain Analysis | "Analyze the trade-offs between aerodynamic downforce limits under Article 3.4.1 and suspension stiffness under Article 10.2." | 55 | 70 | 93 | **94** |
| **MDA-03** | Multi-Domain Analysis (Cross-Doc) | "Compare the impact of the Financial Cost Cap (Article 12.1) on aerodynamic wind tunnel testing hours vs CFD compute allocations." | 45 | 68 | 91 | **92** |
| **CFU-01** | Conversational Follow-up | "Can you summarize that in 3 concise bullet points?" | 92 | 88 | 65 | **95** |