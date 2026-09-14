# College Academic Assistant

An intelligent, lightweight AI agent built for college students to master core academic concepts, query lecture notes using in-memory Retrieval-Augmented Generation (RAG), generate structured study schedules, and test comprehension with tailored quizzes.

Built with **Python**, **Flask**, **Google Gemini**, **LangChain**, and **LangGraph**.

---

## Core Features

- **Academic Question Answering**: Rigorous conceptual explanations powered by Google Gemini and compact domain knowledge.
- **PDF & TXT Notes Upload**: Drag-and-drop or upload lecture notes and textbook excerpts. Files are parsed using `pypdf`, chunked via `RecursiveCharacterTextSplitter`, and embedded into an in-memory vector store.
- **Lightweight In-Memory RAG**: Queries uploaded course material using vector similarity search without requiring external vector databases or disk-backed databases.
- **Built-in Computer Science Knowledge Base**: Instant offline summaries for 7 core subjects:
  - Python
  - Data Structures
  - Database Management Systems (DBMS)
  - Operating Systems
  - Computer Networks
  - Artificial Intelligence
  - Machine Learning
- **Difficult Topic Explanations**: Dedicated intuition-first breakdowns with real-world analogies and code/math examples.
- **Study Plan Generator**: Formulates day-by-day study milestones, active recall topics, and revision goals tailored to exam deadlines and daily hours.
- **Quiz Generator**: Generates multiple-choice questions with answer keys and conceptual explanations across customizable difficulty levels.
- **Conversational Follow-ups**: Maintains dialogue history for iterative clarifications.
- **Strict Academic Integrity**: Distinguishes between `[Uploaded Notes]`, `[Built-in Knowledge Base]`, and `[General Academic Knowledge]`. Strictly avoids fabricating exam dates, faculty info, or university regulations.

---

## Architecture & Workflow

The assistant executes queries through a compiled **LangGraph** state machine:

```
[START]
   │
   ▼
[Understand Question] ─── (Needs Retrieval?) ───► [Retrieve Knowledge]
   │                                                     │
   │ (Direct / No RAG)                                   │
   ▼                                                     ▼
[Generate Answer] ◄──────────────────────────────────────┘
   │
   ▼
 [END]
```

1. **Understand Question**: Analyzes whether the student's inquiry targets uploaded notes, built-in computer science subjects, or general concepts.
2. **Retrieve Knowledge**: Executes similarity search over the in-memory vector store or indexes the built-in knowledge base.
3. **Generate Answer**: Synthesizes the retrieved context and conversation history with Gemini to produce accurate, source-tagged guidance.

---

## Project Structure

```
├── app.py              # Complete application logic (Flask UI, LangGraph, RAG, Tools)
├── requirements.txt    # Python package dependencies
└── README.md           # Project documentation
```

*Note: In accordance with project requirements, `app.py` is under 400 lines (approx. 375 lines), contains zero comments/docstrings, and requires no external database or separate modules.*

---

## Getting Started

### 1. Prerequisites

- Python 3.10 or higher
- A Google Gemini API key (obtainable from [Google AI Studio](https://aistudio.google.com/))

### 2. Clone or Navigate to the Workspace

```bash
cd "college academic assistent"
```

### 3. Set Up Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Key

Create a `.env` file in the root directory (or set the environment variable directly):

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

*(You can also use `GOOGLE_API_KEY=your_gemini_api_key_here`)*

### 6. Run the Application

```bash
python app.py
```

The Flask server will start at:
```
http://localhost:5000
```

Open your web browser and navigate to `http://localhost:5000` to interact with the assistant.
