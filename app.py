import io, os, json
from typing import TypedDict, List, Dict
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
import pypdf
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START, END

load_dotenv()

class SimpleEmbeddings(Embeddings):
    def _vec(self, text):
        v = [0.0] * 128
        for word in text.lower().split(): v[abs(hash(word)) % 128] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]
    def embed_documents(self, texts): return [self._vec(t) for t in texts]
    def embed_query(self, text): return self._vec(text)

def get_embeddings():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        try:
            return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=key)
        except Exception:
            pass
    return SimpleEmbeddings()

def get_llm():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        for m in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                return ChatGoogleGenerativeAI(model=m, google_api_key=key, temperature=0.3)
            except Exception:
                continue
    return None

BUILTIN_KB = {
    "python": "Python: Dynamic typing, GIL, list comprehensions, decorators, generators, cyclic GC.",
    "data structures": "Arrays (O(1)), Linked Lists, Stacks, Queues, Trees, Graphs, Hash Tables.", "dbms": "ACID, Normalization (1NF-BCNF), Indexing (B-trees), Concurrency.",
    "operating systems": "CPU scheduling (FCFS, SJF, RR), Virtual Memory, Paging, Deadlocks, Semaphores.", "computer networks": "OSI & TCP/IP, TCP vs UDP, Subnetting, Routing, DNS, TLS.",
    "artificial intelligence": "A*, Minimax, Alpha-Beta pruning, Heuristics, Propositional & First-Order Logic.", "machine learning": "Supervised (SVM, Random Forest), Unsupervised (K-Means, PCA), Bias-Variance, Neural Nets."
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
vector_store = InMemoryVectorStore(get_embeddings())
uploaded_docs = []

def academic_retriever(query: str) -> dict:
    if uploaded_docs:
        try:
            results = vector_store.similarity_search(query, k=3)
            if results:
                content = "\n\n".join([f"[{r.metadata.get('source', 'Notes')}]: {r.page_content}" for r in results])
                return {"context": content, "source": "Uploaded Notes"}
        except Exception:
            pass
    query_lower = query.lower()
    for topic, info in BUILTIN_KB.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            return {"context": info, "source": "Built-in Knowledge Base"}
    return {"context": "", "source": "General Academic Knowledge"}

SUBJECT_SYLLABUS = {
    "python": ["Syntax & Data Types", "Functions & Scope", "OOP & Inheritance", "Generators & Iterators", "Decorators & Closures", "Memory & Cyclic GC", "Multithreading & GIL"],
    "data structures": ["Arrays & Complexity", "Linked Lists", "Stacks & Queues", "Trees & BST", "Balanced Trees & Heaps", "Hash Tables", "Graphs & Shortest Path"],
    "dbms": ["Relational Model & SQL", "Normalization (1NF-BCNF)", "ACID Transactions", "Concurrency & 2PL", "Indexing & B-Trees", "Recovery & WAL", "Distributed DBMS"],
    "operating systems": ["Processes & Threads", "CPU Scheduling", "Synchronization & Semaphores", "Deadlock Prevention", "Memory Paging", "Virtual Memory", "File Systems"],
    "computer networks": ["OSI & TCP/IP Models", "Data Link & MAC", "Subnetting & Routing", "TCP Handshake vs UDP", "Flow & Congestion Control", "DNS, HTTP & TLS", "Network Security"],
    "artificial intelligence": ["Agents & Environments", "Uninformed Search (BFS/DFS)", "Informed Search & A*", "Minimax & Alpha-Beta", "CSPs", "Knowledge & Logic", "Inference & Resolution"],
    "machine learning": ["Linear/Logistic Regression", "Decision Trees & Random Forests", "SVM & Kernels", "K-Means & PCA", "Bias-Variance Tradeoff", "Gradient Descent", "Neural Networks & Backprop"]
}

SUBJECT_QUIZZES = {
    "python": "1. What is the role of Python's Global Interpreter Lock (GIL)?\nA) Prevents multiple native threads from executing Python bytecodes simultaneously\nB) Accelerates matrix operations in NumPy\nC) Compiles bytecode directly to machine code\nD) Enforces static typing at compile-time\nCorrect: A\nExplanation: The GIL ensures thread-safety by allowing only one native thread to run the interpreter at a time.\n---\n2. How does Python handle circular object references?\nA) Using cyclic garbage collection alongside reference counting\nB) Through immediate heap deallocation\nC) By throwing a MemoryError exception\nD) By converting to weakrefs automatically\nCorrect: A\nExplanation: Cyclic GC runs periodically to detect and collect reference cycles.",
    "data structures": "1. What is the worst-case search time complexity in an unbalanced Binary Search Tree (BST)?\nA) O(1)\nB) O(log n)\nC) O(n)\nD) O(n log n)\nCorrect: C\nExplanation: When elements are inserted in sorted order, an unbalanced BST degenerates into a linear linked list with O(n) search.\n---\n2. Which collision resolution technique places conflicting elements into a linked list at the bucket index?\nA) Separate Chaining\nB) Linear Probing\nC) Quadratic Probing\nD) Double Hashing\nCorrect: A\nExplanation: Separate chaining maintains an auxiliary linked list at each bucket for colliding keys.",
    "dbms": "1. Which ACID property guarantees that all operations within a transaction succeed or all are rolled back?\nA) Atomicity\nB) Consistency\nC) Isolation\nD) Durability\nCorrect: A\nExplanation: Atomicity ensures the all-or-nothing execution of database transaction operations.\n---\n2. A relational table is in Third Normal Form (3NF) if it is in 2NF and has:\nA) No transitive functional dependencies on candidate keys\nB) No multi-valued dependencies\nC) At least three primary keys defined\nD) No foreign keys\nCorrect: A\nExplanation: 3NF eliminates transitive dependencies (X -> Y and Y -> Z where Z is non-prime).",
    "operating systems": "1. Which condition is NOT one of Coffman's four necessary conditions for deadlock?\nA) Preemption allowed\nB) Mutual Exclusion\nC) Hold and Wait\nD) Circular Wait\nCorrect: A\nExplanation: Deadlock requires 'No Preemption'. If preemption is allowed, deadlock cannot persist.\n---\n2. What is the primary cause of 'Thrashing' in virtual memory?\nA) Processes spend more time paging in/out than executing instructions\nB) CPU clock frequency drops dynamically\nC) Deadlock between device drivers\nD) Insufficient swap space on physical drive\nCorrect: A\nExplanation: Thrashing occurs when total working set sizes exceed physical memory, causing continuous page faults.",
    "computer networks": "1. During the standard TCP three-way handshake, what flags are sent by the server in step 2?\nA) SYN-ACK\nB) SYN only\nC) ACK only\nD) FIN-ACK\nCorrect: A\nExplanation: The client initiates with SYN, the server responds with SYN-ACK, and client finishes with ACK.\n---\n2. Which transport protocol provides reliable, ordered, and connection-oriented byte stream delivery?\nA) TCP\nB) UDP\nC) IP\nD) ICMP\nCorrect: A\nExplanation: TCP incorporates sequence numbers, checksums, flow control, and retransmission for reliable delivery.",
    "artificial intelligence": "1. What property must a heuristic h(n) satisfy to guarantee that A* tree search is optimal?\nA) Admissibility (never overestimates true cost to goal)\nB) Strict monotonicity only\nC) Non-zero constant slope\nD) Path history independence\nCorrect: A\nExplanation: An admissible heuristic never overestimates the actual cost to reach the goal.\n---\n2. In Minimax with Alpha-Beta pruning, what does the Beta value represent?\nA) The minimum score the minimizing player is guaranteed\nB) The maximum possible heuristic evaluation\nC) The maximum score the maximizing player is guaranteed\nD) The depth limit of search\nCorrect: A\nExplanation: Alpha is the best score guaranteed to Maximizer, and Beta is the lowest score guaranteed to Minimizer.",
    "machine learning": "1. What is the consequence of high model variance in machine learning?\nA) Overfitting to training data and poor test generalization\nB) High systematic error on both train and test sets\nC) Underfitting due to excessive simplicity\nD) Zero loss on all unseen distributions\nCorrect: A\nExplanation: High variance means the model is overly sensitive to training noise, leading to overfitting.\n---\n2. In classification, what does the 'Precision' metric measure?\nA) Ratio of True Positives to all predicted Positives (TP / (TP + FP))\nB) Ratio of True Positives to all actual Positives (TP / (TP + FN))\nC) Ratio of correct predictions to total predictions\nD) Harmonic mean of True Negatives and False Positives\nCorrect: A\nExplanation: Precision measures exactness: of all samples predicted positive, what proportion was actually positive."
}

def study_plan_generator(subject: str, days: int, hours_per_day: float, notes_context: str = "") -> str:
    llm = get_llm()
    prompt = f"Create an academic study plan for '{subject}' over {days} days with {hours_per_day} hours/day. Context: {notes_context}\nInclude daily milestones, active recall topics, and revision goals. Do not fabricate exam dates or university syllabus."
    if llm:
        try:
            return llm.invoke([HumanMessage(content=prompt)]).content
        except Exception:
            pass
    days = max(1, min(days, 30))
    sub_key = subject.lower().strip()
    milestones = None
    for k, syl in SUBJECT_SYLLABUS.items():
        if k in sub_key or sub_key in k:
            milestones = syl
            break
    if not milestones:
        milestones = [f"Foundations & Key Terminology in {subject}", f"Theoretical Principles & Core Concepts", f"Applied Problem-Solving & Case Studies", f"Advanced Techniques & Edge Cases", f"Synthesis, Mock Testing & Review"]
    plan = [f"Academic Study Plan for {subject} ({days} Days, {hours_per_day} hrs/day):\n"]
    m_len = len(milestones)
    for d in range(1, days):
        topic_day = milestones[(d - 1) % m_len]
        plan.append(f"Day {d}: {topic_day} - {hours_per_day * 0.6:.1f}h theory/reading, {hours_per_day * 0.4:.1f}h practice problems & recall.")
    plan.append(f"Day {days}: Comprehensive mock review, synthesis & self-testing across all concepts.")
    return "\n".join(plan)

def quiz_generator(topic: str, num_questions: int = 3, difficulty: str = "Medium", notes_context: str = "") -> str:
    llm = get_llm()
    prompt = f"Generate {num_questions} multiple-choice quiz questions on '{topic}' at {difficulty} difficulty. Context: {notes_context}\nFormat each question clearly with: Question, Options A/B/C/D, Correct Answer, and Brief Explanation. Separate each question with ---"
    if llm:
        try:
            return llm.invoke([HumanMessage(content=prompt)]).content
        except Exception:
            pass
    top_key = topic.lower().strip()
    for k, q_str in SUBJECT_QUIZZES.items():
        if k in top_key or top_key in k:
            return q_str
    return f"1. In {topic}, what is the foundational principle underpinning core operations?\nA) Formal definition and verified axiomatic rules\nB) Heuristic guessing without proof\nC) Random runtime approximations\nD) Unchecked recursion without base condition\nCorrect: A\nExplanation: Sound domain theory relies on precise formal definitions and verified rules.\n---\n2. When evaluating trade-offs in {topic}, which factor is critical to analyze?\nA) Space and time computational complexity\nB) Ignoring edge case constraints\nC) Arbitrary constant factors\nD) Disregarding memory overhead\nCorrect: A\nExplanation: Algorithmic and theoretical analysis always examines time/space efficiency under boundary conditions."

class AgentState(TypedDict, total=False):
    question: str; mode: str; needs_retrieval: bool; retrieved_context: str; context_source: str; history: List[Dict[str, str]]; answer: str

def understand_question(state: AgentState) -> dict:
    q = state.get("question", "").lower()
    needs = bool(uploaded_docs) or any(k in q for k in BUILTIN_KB) or "note" in q or "file" in q or state.get("mode") == "notes"
    return {"needs_retrieval": needs}

def route_after_understand(state: AgentState) -> str:
    return "retrieve_knowledge" if state.get("needs_retrieval") else "generate_answer"

def retrieve_knowledge(state: AgentState) -> dict:
    retrieved = academic_retriever(state.get("question", ""))
    return {"retrieved_context": retrieved["context"], "context_source": retrieved["source"]}

def generate_answer(state: AgentState) -> dict:
    llm = get_llm()
    ctx = state.get("retrieved_context", "")
    src = state.get("context_source", "General Academic Knowledge")
    q = state.get("question", "")
    mode = state.get("mode", "qa")
    sys_prompt = f"You are a College Academic Assistant. Help students learn concepts and explain topics. Rules: Never fabricate regulations, exam schedules, faculty data, or syllabus. Cite source category as [{src}]. Break topics into intuition, theory, and examples."
    user_msg = f"Mode: {mode}\nContext:\n{ctx}\n\nStudent Question: {q}"
    if llm:
        messages = [SystemMessage(content=sys_prompt)]
        for h in state.get("history", [])[-4:]:
            if h.get("role") == "user":
                messages.append(HumanMessage(content=h.get("text", "")))
            elif h.get("role") == "assistant":
                messages.append(AIMessage(content=h.get("text", "")))
        messages.append(HumanMessage(content=user_msg))
        try:
            ans = llm.invoke(messages).content
            return {"answer": f"**Source: {src}**\n\n{ans}"}
        except Exception as err:
            return {"answer": f"**Source: {src}**\n\n(AI service notice: {err})\n\nContext summary: {ctx or 'No local notes found.'}"}
    ans = f"**Source: {src}**\n\nAcademic overview for '{q}':\n" + (ctx if ctx else "Configure GEMINI_API_KEY in .env for full conversational AI generation.")
    return {"answer": ans}

builder = StateGraph(AgentState)
for n, fn in [("understand_question", understand_question), ("retrieve_knowledge", retrieve_knowledge), ("generate_answer", generate_answer)]: builder.add_node(n, fn)
builder.add_edge(START, "understand_question")
builder.add_conditional_edges("understand_question", route_after_understand, {"retrieve_knowledge": "retrieve_knowledge", "generate_answer": "generate_answer"})
builder.add_edge("retrieve_knowledge", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "Valid file required"}), 400
    file = request.files["file"]
    name = file.filename
    ext = name.rsplit(".", 1)[-1].lower()
    try:
        if ext == "pdf":
            reader = pypdf.PdfReader(io.BytesIO(file.read()))
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
        elif ext == "txt":
            text = file.read().decode("utf-8", errors="ignore")
        else:
            return jsonify({"error": "Only PDF and TXT allowed"}), 400
        if not text.strip():
            return jsonify({"error": "No extractable text found. Scanned or image-only PDFs require OCR."}), 400
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        if chunks:
            vector_store.add_texts(chunks, metadatas=[{"source": name}] * len(chunks))
            uploaded_docs.append({"name": name, "chunks": len(chunks)})
        return jsonify({"message": f"Successfully indexed {len(chunks)} chunks from {name}", "total_docs": len(uploaded_docs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    q = data.get("question", "").strip()
    if not q:
        return jsonify({"error": "Question cannot be empty"}), 400
    state = {"question": q, "mode": data.get("mode", "qa"), "needs_retrieval": False, "retrieved_context": "", "context_source": "", "history": data.get("history", []), "answer": ""}
    result = graph.invoke(state)
    return jsonify({"answer": result.get("answer", ""), "source": result.get("context_source", "")})

@app.route("/api/plan", methods=["POST"])
def plan():
    data = request.get_json() or {}
    subject = data.get("subject", "Computer Science")
    days = int(data.get("days", 7))
    hours = float(data.get("hours", 2.0))
    retrieved = academic_retriever(subject)
    res = study_plan_generator(subject, days, hours, retrieved.get("context", ""))
    return jsonify({"plan": res, "source": retrieved.get("source", "")})

@app.route("/api/quiz", methods=["POST"])
def quiz():
    data = request.get_json() or {}
    topic = data.get("topic", "Data Structures")
    count = int(data.get("count", 3))
    diff = data.get("difficulty", "Medium")
    retrieved = academic_retriever(topic)
    res = quiz_generator(topic, count, diff, retrieved.get("context", ""))
    return jsonify({"quiz": res, "source": retrieved.get("source", "")})

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "docs": uploaded_docs,
        "total_chunks": sum(d["chunks"] for d in uploaded_docs),
        "has_api_key": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
        "builtin_topics": list(BUILTIN_KB.keys())
    })

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>College Academic Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#0f172a;--card:#1e293b;--border:#334155;--primary:#6366f1;--text:#f8fafc;--muted:#94a3b8;--accent:#38bdf8;}
*{box-sizing:border-box;margin:0;padding:0;font-family:'Plus Jakarta Sans',sans-serif;}
body{background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column;}
header{background:rgba(30,41,59,0.9);border-bottom:1px solid var(--border);padding:1rem 2rem;display:flex;justify-content:space-between;align-items:center;}
.badge{background:rgba(99,102,241,0.2);color:var(--accent);padding:0.25rem 0.6rem;border-radius:999px;font-size:0.75rem;font-weight:600;border:1px solid rgba(56,189,248,0.3);}
nav{display:flex;gap:0.5rem;background:#1e293b;padding:0.35rem;border-radius:0.75rem;border:1px solid var(--border);margin:1rem 2rem 0;}
.tab-btn{background:transparent;border:none;color:var(--muted);padding:0.6rem 1.2rem;border-radius:0.5rem;cursor:pointer;font-weight:600;font-size:0.88rem;}
.tab-btn.active{background:var(--primary);color:#fff;}
main{flex:1;padding:1.5rem 2rem;max-width:1100px;margin:0 auto;width:100%;}
.panel{display:none;background:var(--card);border:1px solid var(--border);border-radius:1rem;padding:1.5rem;box-shadow:0 10px 25px rgba(0,0,0,0.3);}
.panel.active{display:block;}
.chat-box{display:flex;flex-direction:column;gap:0.85rem;height:400px;overflow-y:auto;margin-bottom:1rem;}
.msg{max-width:82%;border-radius:0.75rem;padding:0.85rem 1.1rem;line-height:1.5;font-size:0.92rem;}
.msg.user{align-self:flex-end;background:var(--primary);color:#fff;}
.msg.assistant{align-self:flex-start;background:#0f172a;border:1px solid var(--border);}
.msg .src{font-size:0.72rem;color:var(--accent);font-weight:700;margin-bottom:0.35rem;text-transform:uppercase;}
.input-row{display:flex;gap:0.75rem;}
input,select{background:#0f172a;border:1px solid var(--border);color:var(--text);padding:0.75rem 1rem;border-radius:0.5rem;font-size:0.9rem;outline:none;width:100%;}
input:focus,select:focus{border-color:var(--primary);}
button.btn{background:var(--primary);color:#fff;border:none;padding:0.75rem 1.4rem;border-radius:0.5rem;font-weight:600;cursor:pointer;white-space:nowrap;}
button.btn:hover{background:#4f46e5;}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;}
.status-pill{font-size:0.8rem;color:var(--muted);margin-top:0.6rem;}
.output-card{background:#0f172a;border:1px solid var(--border);border-radius:0.5rem;padding:1.2rem;margin-top:1rem;white-space:pre-wrap;line-height:1.6;font-size:0.9rem;max-height:450px;overflow-y:auto;}
</style>
</head>
<body>
<header>
<div style="font-size:1.2rem;font-weight:700;">🎓 College Academic Assistant</div>
<div style="display:flex;gap:0.5rem;"><span class="badge" id="kbStatus">KB: 7 Core Subjects</span><span class="badge" id="ragStatus">Notes: 0 Chunks</span></div>
</header>
<nav>
<button class="tab-btn active" onclick="switchTab('tab-qa')">Academic Q&A</button>
<button class="tab-btn" onclick="switchTab('tab-upload')">Upload Notes (RAG)</button>
<button class="tab-btn" onclick="switchTab('tab-plan')">Study Planner</button>
<button class="tab-btn" onclick="switchTab('tab-quiz')">Quiz Generator</button>
</nav>
<main>
<div id="tab-qa" class="panel active">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
<h2 style="font-size:1.1rem;font-weight:600;">Ask Academic Questions & Clarifications</h2>
<select id="qaMode" style="width:auto;padding:0.4rem 0.8rem;font-size:0.85rem;"><option value="qa">Concept Q&A</option><option value="explain">Explain Difficult Topic</option><option value="notes">Notes Query</option></select>
</div>
<div class="chat-box" id="chatBox"><div class="msg assistant"><div class="src">System Notice</div>Welcome! Ask questions on Python, Data Structures, DBMS, OS, Networks, AI, ML, or upload lecture notes for verified RAG answers.</div></div>
<div class="input-row">
<input type="text" id="qaInput" placeholder="Ask question or follow-up (e.g. Explain Virtual Memory vs Paging)..." onkeydown="if(event.key==='Enter')askQuestion()">
<button class="btn" id="askBtn" onclick="askQuestion()">Ask</button>
</div>
</div>
<div id="tab-upload" class="panel">
<h2 style="font-size:1.1rem;font-weight:600;margin-bottom:0.75rem;">Upload PDF or TXT Notes</h2>
<p style="color:var(--muted);font-size:0.88rem;margin-bottom:1.2rem;">Uploaded files are chunked, embedded, and stored in-memory for instant academic retrieval.</p>
<div style="display:flex;gap:0.75rem;"><input type="file" id="noteFile" accept=".pdf,.txt"><button class="btn" onclick="uploadNotes()">Upload & Index</button></div>
<div class="status-pill" id="uploadStatus">No files uploaded in current session.</div>
<div class="output-card" id="indexedFilesList" style="display:none;"></div>
</div>
<div id="tab-plan" class="panel">
<h2 style="font-size:1.1rem;font-weight:600;margin-bottom:1rem;">Generate Academic Study Plan</h2>
<div class="grid-2">
<div><label style="font-size:0.8rem;color:var(--muted);display:block;margin-bottom:0.35rem;">Subject / Course</label><input type="text" id="planSubject" value="Operating Systems"></div>
<div><label style="font-size:0.8rem;color:var(--muted);display:block;margin-bottom:0.35rem;">Days Until Exam</label><input type="number" id="planDays" value="7" min="1" max="30"></div>
</div>
<div style="margin-bottom:1rem;"><label style="font-size:0.8rem;color:var(--muted);display:block;margin-bottom:0.35rem;">Daily Study Hours</label><input type="number" id="planHours" value="2.5" step="0.5" min="0.5" max="12"></div>
<button class="btn" onclick="generatePlan()">Generate Study Plan</button>
<div class="output-card" id="planOutput" style="display:none;"></div>
</div>
<div id="tab-quiz" class="panel">
<h2 style="font-size:1.1rem;font-weight:600;margin-bottom:1rem;">Generate Topic Quiz</h2>
<div class="grid-2">
<div><label style="font-size:0.8rem;color:var(--muted);display:block;margin-bottom:0.35rem;">Topic or Subject</label><input type="text" id="quizTopic" value="Computer Networks"></div>
<div><label style="font-size:0.8rem;color:var(--muted);display:block;margin-bottom:0.35rem;">Number of Questions</label><input type="number" id="quizCount" value="3" min="1" max="10"></div>
</div>
<div style="margin-bottom:1rem;"><label style="font-size:0.8rem;color:var(--muted);display:block;margin-bottom:0.35rem;">Difficulty</label>
<select id="quizDiff"><option value="Easy">Easy</option><option value="Medium" selected>Medium</option><option value="Hard">Hard</option></select>
</div>
<button class="btn" onclick="generateQuiz()">Generate Quiz</button>
<div class="output-card" id="quizOutput" style="display:none;"></div>
</div>
</main>
<script>
let history=[];
function switchTab(id){
document.querySelectorAll('.tab-btn').forEach(b=>b.classList.toggle('active',b.getAttribute('onclick').includes(id)));
document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id===id));
}
function appendMsg(role,text,source){
const box=document.getElementById('chatBox'),div=document.createElement('div');
div.className='msg '+role;
div.innerHTML=(source?'<div class="src">'+source+'</div>':'')+text.replace(/\n/g,'<br>');
box.appendChild(div);box.scrollTop=box.scrollHeight;
}
async function askQuestion(){
const input=document.getElementById('qaInput'),q=input.value.trim();
if(!q)return;
const mode=document.getElementById('qaMode').value,btn=document.getElementById('askBtn');
appendMsg('user',q);input.value='';input.disabled=true;btn.textContent='Thinking...';
try{
const res=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,mode:mode,history:history})});
const d=await res.json(),ans=d.answer||d.error||'No answer generated.';
appendMsg('assistant',ans,d.source||'General Academic AI');
history.push({role:'user',text:q},{role:'assistant',text:ans});
}catch(e){appendMsg('assistant','Error communicating with server: '+e,'Error');}
finally{input.disabled=false;btn.textContent='Ask';input.focus();}
}
async function uploadNotes(){
const fileInput=document.getElementById('noteFile');
if(!fileInput.files.length)return alert('Select a PDF or TXT file first.');
const form=new FormData();form.append('file',fileInput.files[0]);
const statusEl=document.getElementById('uploadStatus');statusEl.textContent='Uploading and chunking document...';
try{
const res=await fetch('/api/upload',{method:'POST',body:form}),d=await res.json();
statusEl.textContent=res.ok?d.message:('Upload failed: '+(d.error||'Unknown error'));
if(res.ok)updateStatus();
}catch(e){statusEl.textContent='Error uploading: '+e;}
}
async function generatePlan(){
const out=document.getElementById('planOutput');out.style.display='block';out.textContent='Generating structured study plan...';
try{
const res=await fetch('/api/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({subject:document.getElementById('planSubject').value,days:document.getElementById('planDays').value,hours:document.getElementById('planHours').value})});
const d=await res.json();out.textContent=d.plan||'Failed to generate plan.';
}catch(e){out.textContent='Error: '+e;}
}
async function generateQuiz(){
const out=document.getElementById('quizOutput');out.style.display='block';out.textContent='Crafting quiz questions...';
try{
const res=await fetch('/api/quiz',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic:document.getElementById('quizTopic').value,count:document.getElementById('quizCount').value,difficulty:document.getElementById('quizDiff').value})});
const d=await res.json();out.textContent=d.quiz||'Failed to generate quiz.';
}catch(e){out.textContent='Error: '+e;}
}
async function updateStatus(){
try{
const res=await fetch('/api/status'),d=await res.json();
document.getElementById('ragStatus').textContent='Notes: '+d.total_chunks+' Chunks';
if(d.docs&&d.docs.length){
const list=document.getElementById('indexedFilesList');list.style.display='block';
list.textContent='Indexed Documents:\n'+d.docs.map(x=>'• '+x.name+' ('+x.chunks+' chunks)').join('\n');
}
}catch(e){}
}
updateStatus();
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
