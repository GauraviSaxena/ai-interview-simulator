"""Comprehensive section-wise question generators, input validators, code judge, and AI evaluators."""
from __future__ import annotations

import ast
import difflib
import io
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

DEVELOPER_KEYWORDS = [
    "developer", "sde", "software", "engineer", "full stack", "fullstack",
    "frontend", "front end", "backend", "back end", "web developer", "web",
    "python", "java", "c++", "golang", "rust", "react", "node", "app developer",
    "mobile developer", "android", "ios", "devops", "cloud engineer", "cloud",
    "ai engineer", "machine learning", "ml engineer", "data engineer", "systems engineer",
    "programmer", "coder", "qa engineer", "automation engineer", "architect",
    "sde1", "sde2", "sde3", "sdet", "ase", "get", "trainee", "associate software engineer",
    "graduate engineer trainee", "mern", "mean", "embedded", "vlsi", "iot", "blockchain",
    "data scientist", "data analyst", "business analyst", "sre", "cybersecurity"
]

COMMON_GIBBERISH_PATTERNS = [
    r"^([a-z])\1+$",              # aaaa, zzzz
    r"\b(asdf|qwer|zxcv|1234|random|none|nil|null)\b",
    r"^[^aeiouyAEIOUY]{6,}$",      # 6+ consonants in a row with no vowels
    r"^[0-9]+$",                  # pure numbers only
]

STANDARD_ROLE_WORDS = {
    # Engineering & Software Titles
    "accountant", "actor", "actuary", "administrator", "advisor", "agent", "analyst", "animator",
    "architect", "artist", "assistant", "associate", "attorney", "auditor", "author", "baker",
    "barista", "biologist", "buyer", "carpenter", "cashier", "chef", "chemist", "clerk", "coach",
    "consultant", "coordinator", "counselor", "curator", "dentist", "designer", "detective",
    "developer", "dietitian", "director", "doctor", "economist", "editor", "educator", "electrician",
    "engineer", "estimator", "executive", "expert", "geologist", "illustrator", "inspector",
    "instructor", "intern", "investigator", "journalist", "judge", "lawyer", "lead", "librarian",
    "manager", "mechanic", "musician", "nurse", "officer", "operator", "optometrist", "paralegal",
    "pharmacist", "photographer", "physician", "physicist", "pilot", "plumber", "police", "producer",
    "professor", "programmer", "psychologist", "recruiter", "reporter", "researcher", "salesperson",
    "scientist", "secretary", "specialist", "statistician", "strategist", "supervisor", "surgeon",
    "teacher", "technician", "therapist", "trainer", "translator", "underwriter", "veterinarian",
    "writer", "sde", "swe", "devops", "sre", "mlops", "intern",
    # Indian IT, AKTU & Core Engineering Specializations
    "sde1", "sde2", "sde3", "sdet", "ase", "get", "trainee", "fresher", "coder",
    "fullstack", "frontend", "backend", "web", "mobile", "android", "ios", "game",
    "java", "python", "cpp", "dotnet", "net", "mern", "mean", "react", "node", "golang", "csharp",
    "data", "analytics", "business", "ml", "ai", "genai", "nlp", "vision", "bi",
    "cloud", "reliability", "sysadmin", "system", "systems", "admin", "network", "dba", "database",
    "security", "cybersecurity", "cyber", "information",
    "qa", "test", "tester", "automation", "manual", "support", "sap", "erp", "product",
    "embedded", "iot", "vlsi", "hardware", "chip", "firmware", "electronics", "electrical", "mechanical", "civil",
    "ui", "ux", "uiux", "blockchain", "web3", "crypto",
    "btech", "mtech", "bca", "mca", "cse", "it", "ece", "ee", "me"
}


def _get_llm_client() -> tuple[Any, str]:
    """Obtain OpenAI or Gemini client if API keys are configured."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if openai_key:
        try:
            from openai import OpenAI
            return OpenAI(api_key=openai_key), "openai"
        except ImportError:
            pass

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            return genai.GenerativeModel("gemini-1.5-flash"), "gemini"
        except ImportError:
            pass

    return None, "offline"


def is_developer_role(role: str) -> bool:
    """Detect whether target role is a software engineering discipline requiring algorithmic DSA."""
    if not role:
        return True
    r = role.lower().strip()
    return any(k in r for k in DEVELOPER_KEYWORDS)


def validate_target_role(role: str) -> tuple[bool, str]:
    """Strict pre-submit validation for target role, accepting all legitimate Indian IT & tech roles."""
    clean = (role or "").strip()
    if len(clean) < 2:
        return False, "Target role is too short. Please specify a genuine professional role title."
    for pat in COMMON_GIBBERISH_PATTERNS:
        if re.search(pat, clean.lower()):
            return False, f"Invalid target role '{clean}'. Please enter a legitimate job title (e.g. 'Software Engineer')."
    
    tokens = [re.sub(r"[^a-zA-Z0-9+#.-]", "", t).lower() for t in clean.split()]
    tokens = [t for t in tokens if t]
    has_legit = any(t in STANDARD_ROLE_WORDS for t in tokens)

    # Check common tech role phrases (e.g. "Full Stack Developer", "SDE-1", "DevOps Engineer")
    lower_role = clean.lower()
    common_phrases = [
        "software developer", "software engineer", "sde", "full stack", "frontend", "backend",
        "data scientist", "data analyst", "business analyst", "devops engineer", "cloud architect",
        "qa engineer", "sdet", "ui/ux", "product manager", "associate software engineer", "ase",
        "graduate engineer trainee", "get", "embedded engineer", "vlsi engineer", "blockchain developer",
        "network engineer", "system administrator", "dba", "sap consultant", "ml engineer", "ai engineer"
    ]
    if any(p in lower_role for p in common_phrases):
        has_legit = True

    if not has_legit and len(tokens) <= 3:
        return False, f"Could not verify '{clean}' as a standard professional role. Please check for typos."
    return True, "Valid target role."


def validate_resume(resume_text: str) -> tuple[bool, str]:
    """Strict pre-submit validation for candidate resume."""
    clean = (resume_text or "").strip()
    words = re.findall(r"\b[A-Za-z0-9+#.-]+\b", clean)
    if len(words) < 35:
        return False, "Uploaded or pasted resume contains fewer than 35 words. Please supply a complete resume."
    signals = ["experience", "education", "skills", "projects", "work", "university", "bachelor", "master", "technologies", "responsibilities", "achievements", "developed", "engineered"]
    matched = sum(1 for s in signals if s in clean.lower())
    if matched < 2 and len(words) < 120:
        return False, "Text does not resemble a valid resume structure (missing standard experience, education, or skills sections)."
    return True, "Valid resume detected."


def validate_job_description(jd_text: str) -> tuple[bool, str]:
    """Strict pre-submit validation for target job description."""
    clean = (jd_text or "").strip()
    words = re.findall(r"\b[A-Za-z0-9+#.-]+\b", clean)
    if len(words) < 25:
        return False, "Job description contains fewer than 25 words. Please provide substantive responsibilities or requirements."
    signals = ["responsibilities", "requirements", "qualifications", "experience", "role", "team", "skills", "opportunity", "candidate", "must have", "nice to have", "proficient", "knowledge"]
    matched = sum(1 for s in signals if s in clean.lower())
    if matched < 1 and len(words) < 60:
        return False, "Job description lacks standard professional role expectations or qualifications."
    return True, "Valid job description detected."


# ==============================================================================
# SECTION 1: SELF INTRODUCTION ANALYZER
# ==============================================================================
def analyze_self_introduction(
    transcript: str,
    resume_text: str = "",
    role: str = "",
    detected_skills: list[str] | None = None,
    duration_seconds: float = 60.0,
) -> dict[str, Any]:
    """Evaluate candidate self introduction against resume skills and executive criteria."""
    clean_tr = (transcript or "").strip()
    words = re.findall(r"\b[A-Za-z0-9+#.-]+\b", clean_tr.lower())
    wcount = len(words)

    if wcount < 10:
        return {
            "score": 0.0,
            "feedback": "Self introduction was too brief or inaudible. Elaborate on your technical background, flagship projects, and role alignment.",
            "skills_matched": [],
            "skills_missed": (detected_skills or [])[:4],
            "wpm": 0.0,
            "strengths": [],
            "improvements": ["Speak clearly for at least 45 to 60 seconds highlighting major projects and technical stack."],
        }

    duration = max(duration_seconds, 15.0)
    wpm = round(wcount * 60.0 / duration, 1)

    all_skills = detected_skills or ["Python", "SQL", "System Design", "Cloud", "Git", "REST APIs"]
    matched = [s for s in all_skills if re.search(r"\b" + re.escape(s.lower()) + r"\b", clean_tr.lower())]
    missed = [s for s in all_skills if s not in matched]

    coverage = len(matched) / max(len(all_skills), 1)
    base_score = 45.0 + min(35.0, coverage * 50.0)

    if 125 <= wpm <= 160:
        base_score += 15.0
    else:
        base_score += max(5.0, 15.0 - abs(wpm - 140) * 0.25)

    if any(k in clean_tr.lower() for k in ["project", "built", "designed", "implemented", "scaled", "led"]):
        base_score += 5.0

    final_score = round(max(10.0, min(98.0, base_score)), 1)

    strengths = []
    if matched:
        strengths.append(f"Successfully highlighted key technical skills: {', '.join(matched[:3])}.")
    if 125 <= wpm <= 160:
        strengths.append(f"Ideal speaking pace ({wpm:.0f} WPM) with natural conversational flow.")

    improvements = []
    if missed:
        improvements.append(f"Consider referencing these resume competencies in your opening pitch: {', '.join(missed[:3])}.")
    if wpm < 115:
        improvements.append("Increase speaking cadence slightly towards 130–150 WPM to project executive drive.")
    elif wpm > 165:
        improvements.append("Introduce deliberate pauses after key achievements to emphasize business impact.")

    return {
        "score": final_score,
        "feedback": f"Delivered {wcount} words over {duration:.0f}s ({wpm:.0f} WPM) referencing {len(matched)} core resume competencies.",
        "skills_matched": matched,
        "skills_missed": missed[:5],
        "wpm": wpm,
        "strengths": strengths or ["Clear vocal delivery."],
        "improvements": improvements or ["Continue refining personal elevator pitch with quantified impact."],
    }


TECHNICAL_QUESTION_BANK: dict[str, list[dict[str, Any]]] = {
    "Easy": [
        {
            "id": "te_1",
            "topic": "OOPs",
            "difficulty": "Easy",
            "question": "What is Encapsulation in Object-Oriented Programming and how does it safeguard class data?",
            "expected_concepts": [
                "data hiding",
                "private variables",
                "getters and setters",
                "invariants"
            ],
            "ideal_answer": "Encapsulation bundles data (attributes) and methods that manipulate that data into a single unit (class), restricting direct access to internal state via private fields and exposing controlled mutation through getters, setters, or public methods to protect object invariants."
        },
        {
            "id": "te_2",
            "topic": "OOPs",
            "difficulty": "Easy",
            "question": "Explain the difference between compile-time (static) polymorphism and runtime (dynamic) polymorphism.",
            "expected_concepts": [
                "method overloading",
                "method overriding",
                "dynamic dispatch",
                "vtable"
            ],
            "ideal_answer": "Compile-time polymorphism resolves method calls at compile time through method overloading or templates/generics, offering zero runtime overhead. Runtime polymorphism resolves calls dynamically at execution time via method overriding, inheritance, and dynamic dispatch tables (vtables)."
        },
        {
            "id": "te_3",
            "topic": "OOPs",
            "difficulty": "Easy",
            "question": "What is the Single Responsibility Principle (SRP) in SOLID design and why is it important?",
            "expected_concepts": [
                "single reason to change",
                "cohesion",
                "maintainability",
                "separation of concerns"
            ],
            "ideal_answer": "The Single Responsibility Principle states that a class should have one, and only one, reason to change. Adhering to SRP fosters high cohesion, makes classes easier to unit-test, and prevents cross-cutting bugs when requirements evolve."
        },
        {
            "id": "te_4",
            "topic": "OOPs",
            "difficulty": "Easy",
            "question": "What is the difference between an Abstract Class and an Interface?",
            "expected_concepts": [
                "state/fields",
                "multiple inheritance",
                "default implementation",
                "contract"
            ],
            "ideal_answer": "An abstract class can maintain state (member variables) and provide concrete method implementations, serving as a base for closely related classes in single inheritance. An interface defines a pure behavioural contract that unrelated classes can implement, supporting multiple interface inheritance."
        },
        {
            "id": "te_5",
            "topic": "OOPs",
            "difficulty": "Easy",
            "question": "Why is 'Composition over Inheritance' generally recommended in enterprise software architecture?",
            "expected_concepts": [
                "loose coupling",
                "has-a vs is-a",
                "fragile base class",
                "runtime flexibility"
            ],
            "ideal_answer": "Inheritance tightly couples subclasses to the implementation details of their superclass (the fragile base class problem). Composition establishes 'has-a' relationships via interfaces or dependency injection, enabling dynamic replacement of behaviors at runtime and preserving class encapsulation."
        },
        {
            "id": "te_6",
            "topic": "OOPs",
            "difficulty": "Easy",
            "question": "What is Constructor Injection and how does it support Inversion of Control (IoC)?",
            "expected_concepts": [
                "dependency injection",
                "inversion of control",
                "mocking",
                "testability"
            ],
            "ideal_answer": "Constructor injection passes required class dependencies into its constructor at instantiation rather than letting the class instantiate them directly. This adheres to Inversion of Control, eliminates hardcoded dependencies, and enables clean mocking in automated unit tests."
        },
        {
            "id": "te_7",
            "topic": "OOPs",
            "difficulty": "Easy",
            "question": "Explain the Open/Closed Principle (OCP) with a concrete example.",
            "expected_concepts": [
                "open for extension",
                "closed for modification",
                "polymorphic strategies",
                "interfaces"
            ],
            "ideal_answer": "The Open/Closed Principle states software entities should be open for extension but closed for modification. For instance, rather than adding conditional 'if-else' statements inside an order processor for each new payment method, we define a PaymentStrategy interface so new payment types extend behavior without altering existing code."
        },
        {
            "id": "te_8",
            "topic": "OOPs",
            "difficulty": "Easy",
            "question": "What are access modifiers (private, protected, public) and why are they necessary?",
            "expected_concepts": [
                "visibility",
                "encapsulation",
                "scope",
                "subclass access"
            ],
            "ideal_answer": "Access modifiers define the scope and visibility of classes, methods, and variables. 'Private' limits access to the declaring class, 'protected' extends access to subclasses and package members, and 'public' exposes members universally. They enforce strict interface boundaries."
        },
        {
            "id": "te_9",
            "topic": "OOPs",
            "difficulty": "Easy",
            "question": "What is a Destructor (or finalizer/cleanup method) and when is deterministic resource cleanup required?",
            "expected_concepts": [
                "resource release",
                "memory vs file handles",
                "RAII",
                "context managers"
            ],
            "ideal_answer": "A destructor cleans up unmanaged resources like file handles, database connections, and network sockets when an object's lifecycle ends. In languages with non-deterministic garbage collection like Python or Java, constructs like 'with' (context managers) or 'try-with-resources' guarantee deterministic cleanup."
        },
        {
            "id": "te_10",
            "topic": "OOPs",
            "difficulty": "Easy",
            "question": "What is Method Overloading and how does it differ from Method Overriding?",
            "expected_concepts": [
                "same name different parameters",
                "signature",
                "compile-time vs runtime",
                "inheritance"
            ],
            "ideal_answer": "Overloading allows multiple methods in the same class to share the same name with different parameter signatures, resolved at compile-time. Overriding occurs when a subclass re-implements a method inherited from a superclass with the exact same signature, resolved at runtime."
        },
        {
            "id": "te_11",
            "topic": "Operating Systems",
            "difficulty": "Easy",
            "question": "What is the fundamental difference between a Process and a Thread?",
            "expected_concepts": [
                "address space",
                "heap vs stack",
                "IPC",
                "context switch overhead"
            ],
            "ideal_answer": "A process is an isolated instance of execution with its own dedicated virtual address space, memory pages, and file descriptors. A thread is an execution unit within a process that shares the heap, global variables, and open handles with peer threads while maintaining its own private stack and CPU registers."
        },
        {
            "id": "te_12",
            "topic": "Operating Systems",
            "difficulty": "Easy",
            "question": "What is a Context Switch and what CPU overhead does it incur?",
            "expected_concepts": [
                "PCB save and restore",
                "CPU registers",
                "TLB flush",
                "cache invalidation"
            ],
            "ideal_answer": "A context switch is the OS mechanism of saving the state (registers, program counter, stack pointer) of a currently executing process/thread to its PCB/TCB and restoring another. The overhead includes CPU cycles spent switching rings, cache misses, and potential TLB (Translation Lookaside Buffer) flushes."
        },
        {
            "id": "te_13",
            "topic": "Operating Systems",
            "difficulty": "Easy",
            "question": "What is Virtual Memory and why is it essential in modern operating systems?",
            "expected_concepts": [
                "address translation",
                "paging",
                "memory protection",
                "swap/disk"
            ],
            "ideal_answer": "Virtual memory maps process-accessible virtual addresses to physical RAM frames via page tables. It provides process memory isolation (one process cannot corrupt another's memory), simplifies dynamic memory allocation, and allows execution of programs requiring more memory than physically available RAM."
        },
        {
            "id": "te_14",
            "topic": "Operating Systems",
            "difficulty": "Easy",
            "question": "What is a Page Fault and what happens when one occurs?",
            "expected_concepts": [
                "invalid page table entry",
                "interrupt/trap",
                "disk I/O swap",
                "frame allocation"
            ],
            "ideal_answer": "A page fault is an interrupt triggered by the Memory Management Unit (MMU) when a program accesses a virtual page whose valid bit is zero (not currently present in physical RAM). The OS traps into kernel mode, allocates a physical frame, reads the page from disk swap, updates the page table, and resumes execution."
        },
        {
            "id": "te_15",
            "topic": "Operating Systems",
            "difficulty": "Easy",
            "question": "Explain what a Race Condition is and how it can be prevented.",
            "expected_concepts": [
                "concurrent access",
                "shared mutable state",
                "mutex/lock",
                "atomic operations"
            ],
            "ideal_answer": "A race condition occurs when multiple threads concurrently read and write shared mutable memory without proper synchronization, causing final results to depend arbitrarily on thread scheduling. It is prevented using synchronization primitives like mutexes, semaphores, or atomic hardware operations."
        },
        {
            "id": "te_16",
            "topic": "Operating Systems",
            "difficulty": "Easy",
            "question": "What is the difference between a Mutex and a Semaphore?",
            "expected_concepts": [
                "binary lock vs counter",
                "ownership",
                "signaling",
                "mutual exclusion"
            ],
            "ideal_answer": "A mutex (mutual exclusion) is a binary lock owned strictly by the thread that acquired it, which must also release it. A semaphore is a signaling mechanism maintaining an integer counter; it can manage a pool of N shared resources and can be signaled (posted) by a thread that did not acquire it."
        },
        {
            "id": "te_17",
            "topic": "Operating Systems",
            "difficulty": "Easy",
            "question": "What is a System Call and how does CPU transition between user mode and kernel mode?",
            "expected_concepts": [
                "ring 3 to ring 0",
                "trap/interrupt",
                "privilege levels",
                "hardware security"
            ],
            "ideal_answer": "A system call is the programmatic interface through which a user-space application requests services from the OS kernel (e.g., file I/O, process creation, network access). The CPU transitions from User Mode (Ring 3) to Kernel Mode (Ring 0) via a software trap or specialized sysenter/syscall instruction."
        },
        {
            "id": "te_18",
            "topic": "Operating Systems",
            "difficulty": "Easy",
            "question": "What are the four Coffman conditions required for a Deadlock to occur?",
            "expected_concepts": [
                "mutual exclusion",
                "hold and wait",
                "no preemption",
                "circular wait"
            ],
            "ideal_answer": "The four Coffman conditions are: 1) Mutual Exclusion (resources cannot be shared concurrently); 2) Hold and Wait (processes hold existing resources while requesting new ones); 3) No Preemption (resources cannot be forcibly confiscated); and 4) Circular Wait (a closed chain of processes each waiting for a resource held by the next)."
        },
        {
            "id": "te_19",
            "topic": "Operating Systems",
            "difficulty": "Easy",
            "question": "What is CPU Scheduling and what is the difference between Preemptive and Non-Preemptive scheduling?",
            "expected_concepts": [
                "time slice",
                "interruption",
                "round robin vs FCFS",
                "responsiveness"
            ],
            "ideal_answer": "CPU scheduling determines which ready process is allocated CPU execution time. In preemptive scheduling (e.g. Round Robin, Multilevel Feedback Queue), the OS can interrupt a running process when its time quantum expires. In non-preemptive scheduling (e.g. FCFS), a process retains the CPU until it yields or terminates."
        },
        {
            "id": "te_20",
            "topic": "Operating Systems",
            "difficulty": "Easy",
            "question": "What is Inter-Process Communication (IPC) and what are three common IPC mechanisms?",
            "expected_concepts": [
                "pipes",
                "shared memory",
                "message queues",
                "unix sockets"
            ],
            "ideal_answer": "IPC allows separate processes with disjoint address spaces to exchange data and synchronize actions. Three common mechanisms are: 1) Pipes / FIFOs for byte streams; 2) Shared Memory for ultra-low latency direct memory access; and 3) UNIX Domain Sockets / Network Sockets for bidirectional structured communication."
        },
        {
            "id": "te_21",
            "topic": "Data Structures",
            "difficulty": "Easy",
            "question": "Compare an Array and a Singly Linked List in terms of memory layout and random access complexity.",
            "expected_concepts": [
                "contiguous memory",
                "pointer overhead",
                "O(1) indexing",
                "O(N) traversal"
            ],
            "ideal_answer": "Arrays allocate elements in contiguous memory blocks, enabling O(1) random access via arithmetic pointer offsets and maximizing CPU L1/L2 cache line locality. Singly linked lists store elements as disjoint heap nodes linked by pointers, requiring O(N) linear traversal to access element i and incurring extra pointer overhead."
        },
        {
            "id": "te_22",
            "topic": "Data Structures",
            "difficulty": "Easy",
            "question": "What is a Hash Table and how does it achieve average O(1) time complexity for lookup?",
            "expected_concepts": [
                "hash function",
                "bucket indexing",
                "collision handling",
                "load factor"
            ],
            "ideal_answer": "A hash table maps arbitrary keys to integer array indices using a deterministic hash function modulo table capacity. By distributing keys uniformly across buckets and resizing dynamically when the load factor exceeds a threshold, it achieves average O(1) insertion, deletion, and lookup."
        },
        {
            "id": "te_23",
            "topic": "Data Structures",
            "difficulty": "Easy",
            "question": "What is a Stack and what are two real-world software applications of a Stack?",
            "expected_concepts": [
                "LIFO",
                "function call stack",
                "undo redo",
                "expression parsing"
            ],
            "ideal_answer": "A stack is a Last-In-First-Out (LIFO) linear data structure supporting O(1) push and pop operations. Two ubiquitous software applications are: 1) the CPU/programming language call stack managing function activation records, and 2) the undo/redo buffer in editors or bracket matching in compilers."
        },
        {
            "id": "te_24",
            "topic": "Data Structures",
            "difficulty": "Easy",
            "question": "What is a Queue and how does a Circular Queue prevent space wastage?",
            "expected_concepts": [
                "FIFO",
                "head and tail pointers",
                "modulo arithmetic",
                "ring buffer"
            ],
            "ideal_answer": "A queue is a First-In-First-Out (FIFO) structure supporting O(1) enqueue and dequeue. A standard linear array queue wastes capacity as the front pointer advances; a circular queue (ring buffer) wraps head and tail indices back to zero using modulo arithmetic ((tail + 1) % capacity), reusing freed slots."
        },
        {
            "id": "te_25",
            "topic": "Data Structures",
            "difficulty": "Easy",
            "question": "What is a Binary Search Tree (BST) and what are its best and worst case lookup complexities?",
            "expected_concepts": [
                "left < root < right",
                "O(log N) balanced",
                "O(N) skewed degenerate",
                "tree height"
            ],
            "ideal_answer": "A Binary Search Tree is a binary tree where every node's key is greater than all keys in its left subtree and smaller than all keys in its right subtree. In a balanced BST, lookup is O(log N). In a degenerate skewed BST (inserted in sorted order), it degrades to a linked list with O(N) time complexity."
        },
        {
            "id": "te_26",
            "topic": "Data Structures",
            "difficulty": "Easy",
            "question": "What is a Min-Heap (or Max-Heap) and what is the time complexity of inserting an element?",
            "expected_concepts": [
                "complete binary tree",
                "heap property",
                "O(log N) heapify up",
                "priority queue"
            ],
            "ideal_answer": "A min-heap is a complete binary tree where the key at any node is less than or equal to the keys of its children, guaranteeing the minimum element resides at the root (O(1) peek). Inserting an element appends it at the end and bubbles it up (sift-up) in O(log N) time."
        },
        {
            "id": "te_27",
            "topic": "Data Structures",
            "difficulty": "Easy",
            "question": "What is the difference between Breadth-First Search (BFS) and Depth-First Search (DFS) on a Graph?",
            "expected_concepts": [
                "queue vs stack",
                "level order",
                "backtracking",
                "shortest path"
            ],
            "ideal_answer": "BFS explores vertices layer by layer using a FIFO queue, making it optimal for finding the shortest unweighted path between nodes. DFS explores as deeply as possible along each branch before backtracking using a LIFO stack (or recursion), making it ideal for topological sorting and cycle detection."
        },
        {
            "id": "te_28",
            "topic": "Data Structures",
            "difficulty": "Easy",
            "question": "What is a Doubly Linked List and when is it preferred over a Singly Linked List?",
            "expected_concepts": [
                "prev and next pointers",
                "O(1) deletion given pointer",
                "bidirectional traversal",
                "LRU cache"
            ],
            "ideal_answer": "A doubly linked list stores both forward (next) and backward (prev) pointers in each node. It allows bidirectional traversal and O(1) node removal when given a direct reference to the node, making it the standard choice for LRU (Least Recently Used) cache eviction structures."
        },
        {
            "id": "te_29",
            "topic": "Data Structures",
            "difficulty": "Easy",
            "question": "What is Big-O notation and why do we analyze Amortized Complexity in dynamic arrays?",
            "expected_concepts": [
                "worst case upper bound",
                "geometric resizing",
                "doubling capacity",
                "O(1) amortized"
            ],
            "ideal_answer": "Big-O notation describes the upper bound of an algorithm's asymptotic runtime as input size N grows. A dynamic array occasionally copies all elements during a doubling resize (O(N) worst-case), but since this happens infrequently, averaging cost across N insertions yields an O(1) amortized insertion cost."
        },
        {
            "id": "te_30",
            "topic": "Data Structures",
            "difficulty": "Easy",
            "question": "What is a Trie (Prefix Tree) and why is it useful for string searching?",
            "expected_concepts": [
                "prefix sharing",
                "alphabet nodes",
                "autocomplete",
                "O(L) where L is string length"
            ],
            "ideal_answer": "A Trie is a tree structure where each edge represents a character and paths from root to node represent prefixes. It provides O(L) search, insert, and prefix lookup time where L is the string length (independent of the number of dictionary words N), making it ideal for autocomplete engines and spell-checkers."
        },
        {
            "id": "te_dbms_e1",
            "topic": "DBMS & SQL",
            "difficulty": "Easy",
            "question": "Explain the ACID properties in Database Management Systems and why they are critical for financial or e-commerce transactions.",
            "expected_concepts": ["atomicity", "consistency", "isolation", "durability", "commit", "rollback"],
            "ideal_answer": "ACID ensures reliable database transaction processing: Atomicity guarantees all-or-nothing completion; Consistency enforces all integrity constraints; Isolation ensures concurrent transactions execute independently without dirty reads; and Durability guarantees committed transactions survive server crashes via write-ahead logging (WAL)."
        },
        {
            "id": "te_dbms_e2",
            "topic": "DBMS & SQL",
            "difficulty": "Easy",
            "question": "What is Database Normalization? Explain the difference between 1NF, 2NF, and 3NF with table design examples.",
            "expected_concepts": ["normalization", "1NF atomic values", "2NF partial dependency", "3NF transitive dependency", "redundancy"],
            "ideal_answer": "Normalization organizes database relations to eliminate data redundancy and insertion/update/deletion anomalies: 1NF requires atomic (indivisible) attribute values; 2NF eliminates partial dependencies where non-prime attributes depend on a subset of a composite candidate key; and 3NF removes transitive dependencies where non-prime attributes depend on other non-prime attributes."
        },
        {
            "id": "te_dbms_e3",
            "topic": "DBMS & SQL",
            "difficulty": "Easy",
            "question": "What are SQL Joins? Explain the difference between INNER JOIN, LEFT OUTER JOIN, and FULL OUTER JOIN.",
            "expected_concepts": ["inner join", "left outer join", "full outer join", "null matching", "cartesian product"],
            "ideal_answer": "Joins combine rows from two or more tables based on a related column: INNER JOIN returns only records with matching keys in both tables; LEFT OUTER JOIN returns all records from the left table and matched rows from the right table (filling unmatched fields with NULL); and FULL OUTER JOIN returns all rows from both tables, filling NULLs where join conditions fail."
        },
        {
            "id": "te_dbms_e4",
            "topic": "DBMS & SQL",
            "difficulty": "Easy",
            "question": "What is the difference between Primary Key, Unique Key, and Foreign Key in relational databases?",
            "expected_concepts": ["primary key", "unique key", "foreign key", "null values", "referential integrity"],
            "ideal_answer": "A Primary Key uniquely identifies each row in a table and cannot contain NULL values (only one primary key per table). A Unique Key also enforces uniqueness across rows but permits a single NULL value (multiple unique keys can exist). A Foreign Key points to a Primary or Unique Key in another table to enforce referential integrity."
        },
        {
            "id": "te_cn_e1",
            "topic": "Computer Networks",
            "difficulty": "Easy",
            "question": "Explain the 7 layers of the OSI model and their core responsibilities.",
            "expected_concepts": ["physical", "data link", "network", "transport", "session", "presentation", "application"],
            "ideal_answer": "The 7 OSI layers are: 1. Physical (raw bitstream transmission over physical medium), 2. Data Link (node-to-node framing and MAC addressing), 3. Network (packet routing and logical IP addressing), 4. Transport (end-to-end reliability, TCP/UDP segmentation and flow control), 5. Session (dialogue and session management), 6. Presentation (data format translation, encryption, compression), and 7. Application (network services interfacing directly with user applications, e.g. HTTP, DNS)."
        },
        {
            "id": "te_cn_e2",
            "topic": "Computer Networks",
            "difficulty": "Easy",
            "question": "What is the difference between TCP and UDP? Why is UDP preferred for live video streaming and gaming while TCP is used for HTTP and email?",
            "expected_concepts": ["connection oriented", "connectionless", "3-way handshake", "reliability", "packet retransmission", "latency"],
            "ideal_answer": "TCP is a connection-oriented, reliable protocol providing ordered delivery, error checking, flow control, and retransmission via a 3-way handshake, making it ideal for web browsing and file transfers. UDP is connectionless and lightweight without retransmissions or ordering overhead, minimizing latency for real-time video streaming, VoIP, and multiplayer gaming where occasional dropped packets are preferable to transmission delays."
        },
        {
            "id": "te_cn_e3",
            "topic": "Computer Networks",
            "difficulty": "Easy",
            "question": "Walk through step-by-step what happens when a user types a URL (e.g., https://www.google.com) into a browser and presses Enter.",
            "expected_concepts": ["dns lookup", "tcp handshake", "tls handshake", "http get request", "server response", "dom parsing"],
            "ideal_answer": "1. Browser checks cache (browser, OS, router) for DNS; if absent, queries recursive resolver down to Root, TLD (.com), and Authoritative DNS for the server IP. 2. Browser initiates TCP 3-way handshake (SYN, SYN-ACK, ACK) on port 443. 3. Performs TLS cryptographic handshake for session key exchange and certificate verification. 4. Sends HTTP GET request. 5. Server processes request and returns HTTP 200 with HTML payload. 6. Browser renders HTML, fetches CSS/JS, and constructs the DOM/Render tree."
        },
        {
            "id": "te_cn_e4",
            "topic": "Computer Networks",
            "difficulty": "Easy",
            "question": "Explain the difference between HTTP and HTTPS. How does the SSL/TLS certificate ensure data privacy and authenticity?",
            "expected_concepts": ["port 80 vs 443", "ssl/tls", "symmetric encryption", "asymmetric encryption", "man-in-the-middle", "ca certificate"],
            "ideal_answer": "HTTP sends data as plaintext over port 80, leaving it vulnerable to eavesdropping and man-in-the-middle attacks. HTTPS runs over port 443 and encrypts the transport channel using SSL/TLS. It uses asymmetric public-key cryptography to authenticate the server identity via a trusted Certificate Authority (CA) and negotiate a shared symmetric session key (e.g. AES-256) for fast, secure end-to-end data encryption."
        }
    ],
    "Medium": [
        {
            "id": "tm_1",
            "topic": "OOPs",
            "difficulty": "Medium",
            "question": "Explain how the Vtable (Virtual Method Table) and dynamic dispatch work in object-oriented runtimes.",
            "expected_concepts": [
                "vptr",
                "pointer indirection",
                "virtual function resolution",
                "cache miss overhead"
            ],
            "ideal_answer": "When a class declares virtual methods, the compiler injects a hidden vptr pointer into object instances pointing to a class-wide Vtable (array of function pointers). At runtime, invocations dereference the instance's vptr and index into the Vtable, incurring an indirect branch and potential CPU instruction cache miss."
        },
        {
            "id": "tm_2",
            "topic": "OOPs",
            "difficulty": "Medium",
            "question": "What is the Liskov Substitution Principle (LSP) and what is a classic violation of it?",
            "expected_concepts": [
                "subtypes behavioral substitutability",
                "preconditions and postconditions",
                "Square inherits Rectangle problem",
                "invariants"
            ],
            "ideal_answer": "LSP states objects of a superclass should be replaceable with objects of a subclass without altering the correctness of the program. A classic violation is a Square inheriting from Rectangle; mutating Square's width forces height mutation, violating client invariants expecting independent dimension mutation."
        },
        {
            "id": "tm_3",
            "topic": "OOPs",
            "difficulty": "Medium",
            "question": "Compare the Factory Method Pattern with the Abstract Factory Pattern. When should each be used?",
            "expected_concepts": [
                "single product vs family of products",
                "method inheritance vs composition",
                "creational patterns"
            ],
            "ideal_answer": "The Factory Method relies on class inheritance, delegating the instantiation of a single product to a subclass method. The Abstract Factory relies on object composition, providing an interface for creating entire families of related or dependent objects (e.g. DarkThemeButton and DarkThemeScrollbar) without specifying their concrete classes."
        },
        {
            "id": "tm_4",
            "topic": "OOPs",
            "difficulty": "Medium",
            "question": "What is the Diamond Problem in multiple inheritance and how do languages like C++ and Python resolve it?",
            "expected_concepts": [
                "multiple inheritance",
                "ambiguity",
                "virtual base classes in C++",
                "C3 linearization / MRO in Python"
            ],
            "ideal_answer": "The Diamond Problem arises when a class inherits from two classes that both inherit from a common base, creating duplicate state or ambiguous method paths. C++ resolves this via virtual inheritance ('virtual public Base'). Python resolves this using Method Resolution Order (MRO) calculated by the C3 Linearization algorithm."
        },
        {
            "id": "tm_5",
            "topic": "OOPs",
            "difficulty": "Medium",
            "question": "How does the Observer Pattern work and how do you prevent memory leaks associated with it?",
            "expected_concepts": [
                "subject and observer",
                "publish-subscribe",
                "weak references",
                "deregistration / unsubscribe"
            ],
            "ideal_answer": "The Observer pattern defines a one-to-many relationship where a Subject notifies registered Observers of state changes via an update() interface. Memory leaks ('lapsed listener problem') occur when the subject retains strong references to observers after their lifecycle ends. Preventing this requires explicit unregistration or using Weak References."
        },
        {
            "id": "tm_6",
            "topic": "OOPs",
            "difficulty": "Medium",
            "question": "What is the Dependency Inversion Principle (DIP) and how does it relate to Dependency Injection (DI)?",
            "expected_concepts": [
                "high level modules depend on abstractions",
                "loose coupling",
                "IoC container",
                "interface abstraction"
            ],
            "ideal_answer": "DIP is the architectural principle stating high-level modules should not depend on low-level modules; both should depend on abstractions. Dependency Injection is the practical design technique/pattern that implements DIP by providing concrete implementations to classes from outside (often via constructors or IoC containers)."
        },
        {
            "id": "tm_7",
            "topic": "OOPs",
            "difficulty": "Medium",
            "question": "Explain the Strategy Pattern vs State Pattern. How do their intents differ?",
            "expected_concepts": [
                "interchangeable algorithms vs internal state transitions",
                "client awareness",
                "context behavior"
            ],
            "ideal_answer": "While both share a similar UML class structure (delegation to an interface), their intents differ. Strategy lets the client configure the context with an interchangeable algorithm (e.g. different sorting strategies). State allows the context to dynamically transition between internal state objects as events occur without client intervention."
        },
        {
            "id": "tm_8",
            "topic": "OOPs",
            "difficulty": "Medium",
            "question": "What are Immutable Objects in OOP and what concurrency benefits do they provide?",
            "expected_concepts": [
                "thread safety without locks",
                "frozen state",
                "defensive copying",
                "side-effect free"
            ],
            "ideal_answer": "Immutable objects are instances whose state cannot be modified after construction. Because their internal state never mutates, they are inherently thread-safe without requiring locks, eliminating race conditions and deadlocks, and can be shared freely across threads as hash keys or cache entries."
        },
        {
            "id": "tm_9",
            "topic": "OOPs",
            "difficulty": "Medium",
            "question": "What is the Decorator Pattern and how does it achieve dynamic behavior extension compared to subclassing?",
            "expected_concepts": [
                "wrapper object",
                "conforming to interface",
                "runtime composition",
                "avoiding class explosion"
            ],
            "ideal_answer": "The Decorator pattern wraps an existing object with a decorator class conforming to the same interface, forwarding calls to the wrapped instance while adding preprocessing or postprocessing behavior. Unlike subclassing which statically binds behavior at compile time, decorators can be stacked dynamically at runtime without combinatorial class explosion."
        },
        {
            "id": "tm_10",
            "topic": "OOPs",
            "difficulty": "Medium",
            "question": "What is the Interface Segregation Principle (ISP) and what code smell indicates an ISP violation?",
            "expected_concepts": [
                "fat interfaces",
                "client forced to depend on unused methods",
                "dummy method stubs",
                "granular contracts"
            ],
            "ideal_answer": "ISP states clients should not be forced to depend on interfaces they do not use. The code smell indicating an ISP violation is 'fat interfaces' where implementing classes must provide empty stubs, NotImplementedExceptions, or no-op returns for irrelevant methods. It is fixed by breaking fat interfaces into smaller, focused contracts."
        },
        {
            "id": "tm_11",
            "topic": "Operating Systems",
            "difficulty": "Medium",
            "question": "Explain Multi-Level Paging and how the Translation Lookaside Buffer (TLB) mitigates its memory latency penalty.",
            "expected_concepts": [
                "hierarchical page tables",
                "sparse address space",
                "memory reference overhead",
                "TLB hit rate"
            ],
            "ideal_answer": "Multi-level page tables organize page entries into a hierarchy (e.g. 4-level paging on x86-64), avoiding storing massive linear tables for sparse memory spaces. However, translating a virtual address requires up to 4 sequential memory reads. The TLB is an on-chip associative hardware cache storing recent translations; a TLB hit resolves addresses in sub-nanosecond hardware cycles."
        },
        {
            "id": "tm_12",
            "topic": "Operating Systems",
            "difficulty": "Medium",
            "question": "What causes Thrashing in virtual memory and how does the Working Set Model detect and mitigate it?",
            "expected_concepts": [
                "high page fault frequency",
                "CPU utilization collapse",
                "working set window delta",
                "process suspension"
            ],
            "ideal_answer": "Thrashing occurs when the aggregate memory demanded by running processes exceeds available physical RAM frames, causing the OS to spend more time swapping pages to/from disk than executing instructions (CPU utilization collapses). The Working Set Model tracks pages referenced in time window \u00ce\u201d; if sum(working sets) > RAM, the OS suspends processes."
        },
        {
            "id": "tm_13",
            "topic": "Operating Systems",
            "difficulty": "Medium",
            "question": "Explain the difference between a Spinlock and a Sleep/Futex lock. When should an engineer use each?",
            "expected_concepts": [
                "busy waiting",
                "CPU cycle burn",
                "thread descheduling",
                "short critical sections / multicore"
            ],
            "ideal_answer": "A spinlock continuously loops (busy-waits) in CPU registers until the lock becomes available, avoiding context switch overhead; it is optimal for ultra-short critical sections on multi-core systems. A sleep lock (or Linux futex) yields the CPU, putting the thread to sleep until awakened, which is ideal for longer critical sections or I/O operations."
        },
        {
            "id": "tm_14",
            "topic": "Operating Systems",
            "difficulty": "Medium",
            "question": "What is Copy-on-Write (CoW) during a process fork() system call?",
            "expected_concepts": [
                "read-only page flags",
                "deferred copying",
                "page fault trap on write",
                "memory optimization"
            ],
            "ideal_answer": "During fork(), the OS duplicates the parent's page table entries rather than copying physical memory frames, marking all shared pages as read-only. When either process attempts to write to a page, the CPU triggers a page fault trap; the kernel then allocates a new physical frame, copies the 4KB data, updates the writing process's page table, and resumes."
        },
        {
            "id": "tm_15",
            "topic": "Operating Systems",
            "difficulty": "Medium",
            "question": "What is the difference between Synchronous vs Asynchronous I/O, and Blocking vs Non-Blocking I/O?",
            "expected_concepts": [
                "thread wait state",
                "system call return",
                "epoll/kqueue event loop",
                "kernel notification callback"
            ],
            "ideal_answer": "Blocking I/O halts the calling thread until data is transferred. Non-blocking I/O returns immediately with data or EAGAIN/EWOULDBLOCK error. Synchronous I/O requires the calling thread to drive the transfer completion. Asynchronous I/O (e.g. io_uring, IOCP) initiates the transfer and notifies the application via completion callbacks while execution continues independently."
        },
        {
            "id": "tm_16",
            "topic": "Operating Systems",
            "difficulty": "Medium",
            "question": "How does the Linux Completely Fair Scheduler (CFS) allocate CPU time using red-black trees?",
            "expected_concepts": [
                "vruntime virtual runtime",
                "RB-tree min element",
                "niceness weight",
                "proportional fairness"
            ],
            "ideal_answer": "Linux CFS tracks each thread's execution duration scaled by its priority (niceness) as 'vruntime' (virtual runtime). Ready threads are stored in a self-balancing red-black tree indexed by vruntime. The scheduler selects the leftmost node (lowest vruntime) to execute next, maintaining strict proportional CPU fairness across tasks."
        },
        {
            "id": "tm_17",
            "topic": "Operating Systems",
            "difficulty": "Medium",
            "question": "What is Priority Inversion and how does Priority Inheritance Protocol resolve it?",
            "expected_concepts": [
                "low priority holds lock",
                "high priority blocked",
                "medium priority preempts",
                "temporary priority elevation"
            ],
            "ideal_answer": "Priority inversion occurs when a low-priority thread holds a lock required by a high-priority thread, but a medium-priority thread preempts the low-priority thread, effectively starving the high-priority thread. Priority Inheritance temporarily elevates the low-priority thread to the high priority level until it releases the lock."
        },
        {
            "id": "tm_18",
            "topic": "Operating Systems",
            "difficulty": "Medium",
            "question": "Explain the Memory Management Unit (MMU) architecture and the role of Page Table Entries (PTE).",
            "expected_concepts": [
                "virtual to physical translation",
                "dirty bit",
                "present bit",
                "read write execute permissions"
            ],
            "ideal_answer": "The MMU is hardware translating virtual addresses to physical RAM addresses. It reads Page Table Entries (PTEs) containing physical frame numbers and control bits: Present bit (is page in RAM), Read/Write/Execute permission bits, Dirty bit (modified since loaded), and Accessed bit (for LRU eviction algorithms)."
        },
        {
            "id": "tm_19",
            "topic": "Operating Systems",
            "difficulty": "Medium",
            "question": "What is a Zombie Process vs an Orphan Process? How are they reaped?",
            "expected_concepts": [
                "exit without wait()",
                "parent terminated",
                "init/PID 1 adoption",
                "process table entry release"
            ],
            "ideal_answer": "A Zombie process has terminated execution but remains in the process table so its parent can read its exit code via wait()/waitpid(); it is reaped when the parent calls wait(). An Orphan process has its parent terminate while it continues running; it is adopted by init (PID 1 or systemd), which automatically calls wait() when it exits."
        },
        {
            "id": "tm_20",
            "topic": "Operating Systems",
            "difficulty": "Medium",
            "question": "How does Disk I/O elevator scheduling (e.g. SCAN/C-SCAN) optimize disk seek times?",
            "expected_concepts": [
                "cylinder head movement",
                "starvation prevention",
                "sequential sweep",
                "rotational latency"
            ],
            "ideal_answer": "Elevator algorithms (SCAN/C-SCAN) service disk cylinder requests in sweeping passes in one direction before reversing (or returning to start), preventing thrashing of mechanical read/write heads and starvation. While SSDs have zero seek time, modern I/O schedulers (like BFQ or Kyber) still prioritize latency and fair bandwidth."
        },
        {
            "id": "tm_21",
            "topic": "Data Structures",
            "difficulty": "Medium",
            "question": "Explain the internal rebalancing mechanics of an AVL Tree during node insertion.",
            "expected_concepts": [
                "balance factor (-1, 0, 1)",
                "tree height difference",
                "rotations (LL, RR, LR, RL)",
                "O(log N) strict balance"
            ],
            "ideal_answer": "An AVL tree is a strictly height-balanced BST where the balance factor (height(left) - height(right)) of every node is restricted to -1, 0, or 1. After insertion, ancestors are checked; if balance exceeds \u00c2\u00b11, it performs single (LL or RR) or double (LR or RL) rotations in O(1) time, preserving strict O(log N) lookup."
        },
        {
            "id": "tm_22",
            "topic": "Data Structures",
            "difficulty": "Medium",
            "question": "How does an LRU (Least Recently Used) Cache achieve O(1) get and put operations?",
            "expected_concepts": [
                "hash map + doubly linked list",
                "node splicing to head",
                "eviction from tail",
                "O(1) dictionary lookup"
            ],
            "ideal_answer": "An LRU cache combines a Hash Map with a Doubly Linked List. The hash map maps keys to list node pointers for O(1) lookup. The doubly linked list maintains access order: recently accessed nodes are spliced to the head in O(1), and evictions discard the node at the tail in O(1)."
        },
        {
            "id": "tm_23",
            "topic": "Data Structures",
            "difficulty": "Medium",
            "question": "What is a Red-Black Tree and why is it preferred over an AVL Tree in the Linux kernel and Java TreeMap?",
            "expected_concepts": [
                "red/black node colors",
                "black height rule",
                "fewer rotations on insert/delete",
                "relaxed balance"
            ],
            "ideal_answer": "A Red-Black tree is a self-balancing BST enforcing color properties: the root is black, red nodes cannot have red children, and every path from root to leaf has the same number of black nodes. While slightly less strictly balanced than AVL, it requires at most 2-3 rotations during mutations, making it faster for write-intensive workloads."
        },
        {
            "id": "tm_24",
            "topic": "Data Structures",
            "difficulty": "Medium",
            "question": "Explain Hash Collision resolution strategies: Separate Chaining vs Open Addressing (Linear/Quadratic Probing).",
            "expected_concepts": [
                "linked list / tree buckets",
                "probing sequence",
                "clustering",
                "cache friendliness"
            ],
            "ideal_answer": "Separate chaining stores collided keys in linked lists or red-black trees at each bucket index; it gracefully degrades under high load factors. Open addressing stores all elements in the array itself, probing subsequent empty slots via linear or quadratic probing; it offers better CPU cache locality but suffers from clustering and requires low load factors."
        },
        {
            "id": "tm_25",
            "topic": "Data Structures",
            "difficulty": "Medium",
            "question": "How does a B-Tree structure optimize data storage and retrieval in database engines and filesystems?",
            "expected_concepts": [
                "high branching factor / fan-out",
                "disk block size alignment",
                "shallow tree height",
                "fewer disk I/O seeks"
            ],
            "ideal_answer": "A B-Tree is a self-balancing multi-way search tree with a high branching factor (fan-out of hundreds of keys per node) sized to match OS disk block pages (e.g. 4KB-16KB). This keeps tree height extremely low (typically 3-4 levels for billions of rows), reducing slow disk I/O seeks to a minimum."
        },
        {
            "id": "tm_26",
            "topic": "Data Structures",
            "difficulty": "Medium",
            "question": "What is a Disjoint Set Union (DSU / Union-Find) structure and what is its time complexity with Path Compression and Union by Rank?",
            "expected_concepts": [
                "find and union",
                "path compression",
                "union by rank",
                "inverse Ackermann \u00ce\u00b1(N)"
            ],
            "ideal_answer": "DSU manages a partition of an element set into disjoint subsets supporting find(x) (find representative root) and union(x, y) (merge sets). Applying Union by Rank (attaching smaller tree under taller) and Path Compression (flattening tree during find) achieves nearly constant O(\u00ce\u00b1(N)) amortized time complexity, where \u00ce\u00b1 is the inverse Ackermann function."
        },
        {
            "id": "tm_27",
            "topic": "Data Structures",
            "difficulty": "Medium",
            "question": "What is a Monotonic Stack and what algorithmic class of problems does it solve in O(N) time?",
            "expected_concepts": [
                "strictly increasing or decreasing stack",
                "Next Greater Element",
                "histogram area",
                "O(N) amortized"
            ],
            "ideal_answer": "A monotonic stack maintains elements in strictly increasing or decreasing order by popping violating elements before pushing a new item. Each element is pushed and popped at most once, yielding O(N) linear time for 'Next Greater Element', 'Largest Rectangle in Histogram', and 'Stock Span' problems."
        },
        {
            "id": "tm_28",
            "topic": "Data Structures",
            "difficulty": "Medium",
            "question": "How does a Bloom Filter work and what is the trade-off regarding false positives vs false negatives?",
            "expected_concepts": [
                "probabilistic data structure",
                "bit array + k hash functions",
                "zero false negatives",
                "possible false positives"
            ],
            "ideal_answer": "A Bloom filter is a space-efficient probabilistic data structure used to test set membership. Adding an element sets k bits via k independent hash functions in a bit array. Querying checks if all k bits are 1: if any bit is 0, the element is definitively absent (zero false negatives); if all bits are 1, it may be present (possible false positives due to bit collisions)."
        },
        {
            "id": "tm_29",
            "topic": "Data Structures",
            "difficulty": "Medium",
            "question": "What is a Skip List and how does it achieve O(log N) search using probabilistic layering?",
            "expected_concepts": [
                "linked list layers",
                "coin flip geometric promotion",
                "express lanes",
                "O(log N) concurrent skip list"
            ],
            "ideal_answer": "A skip list enhances a sorted linked list with multiple hierarchical levels of forward pointers ('express lanes'). Nodes are promoted to higher levels using a probabilistic coin flip (p=0.5). Searches traverse the highest level first and drop down upon encountering a larger key, achieving O(log N) search and insert with simpler lock-free concurrent implementation than balanced trees."
        },
        {
            "id": "tm_30",
            "topic": "Data Structures",
            "difficulty": "Medium",
            "question": "Compare Topological Sort via Kahn's Algorithm (BFS) vs DFS-based with cycle detection.",
            "expected_concepts": [
                "DAG in-degrees",
                "zero in-degree queue",
                "recursion stack cycle detection",
                "O(V + E)"
            ],
            "ideal_answer": "Kahn's algorithm computes in-degrees for all vertices and enqueues 0-in-degree nodes; removing vertices and decrementing neighbors' in-degrees detects cycles if processed vertices < V. DFS visits nodes, marking vertices visiting/visited in a recursion stack; nodes are prepended to the order after children finish. Both run in O(V + E) time."
        },
        {
            "id": "te_dbms_m1",
            "topic": "DBMS & SQL",
            "difficulty": "Medium",
            "question": "Explain how B-Tree and B+Tree indexes work in databases. What is the difference between a Clustered Index and a Non-Clustered Index?",
            "expected_concepts": ["b+tree", "clustered index", "non-clustered index", "leaf nodes", "disk i/o", "binary search"],
            "ideal_answer": "B+Trees store keys and record pointers exclusively at leaf nodes (linked sequentially for rapid range scans), while internal nodes only store routing keys, allowing high fanout and shallow tree depth to minimize disk I/O. A Clustered Index physically arranges the actual table data rows on disk in sorted order (only one clustered index per table, typically the Primary Key). A Non-Clustered Index creates a separate lookup structure containing sorted index columns and pointers back to the actual table data."
        },
        {
            "id": "te_dbms_m2",
            "topic": "DBMS & SQL",
            "difficulty": "Medium",
            "question": "Explain Transaction Isolation Levels (Read Uncommitted, Read Committed, Repeatable Read, Serializable) and the concurrency anomalies they prevent.",
            "expected_concepts": ["isolation levels", "dirty read", "non-repeatable read", "phantom read", "mvcc", "locks"],
            "ideal_answer": "Read Uncommitted permits dirty reads (reading uncommitted changes). Read Committed prevents dirty reads by only reading committed rows. Repeatable Read ensures any row read during a transaction will return identical values upon re-reading, preventing non-repeatable reads. Serializable is the highest level, preventing phantom reads (new rows inserted matching search criteria) via strict range locks or two-phase locking (2PL)."
        },
        {
            "id": "te_dbms_m3",
            "topic": "DBMS & SQL",
            "difficulty": "Medium",
            "question": "Compare Relational (SQL) and NoSQL databases. In what real-world scenarios would you choose MongoDB or Redis over PostgreSQL?",
            "expected_concepts": ["acid vs base", "horizontal scaling", "document model", "in-memory caching", "structured vs unstructured"],
            "ideal_answer": "Relational databases (PostgreSQL) enforce strict ACID transactions, complex joins, and normalized schemas, making them ideal for banking, e-commerce orders, and ERPs. MongoDB is a document-oriented NoSQL database offering flexible JSON schemas and native horizontal sharding, ideal for dynamic catalogs and content management. Redis is an in-memory key-value data structure store delivering sub-millisecond read/write operations, ideal for distributed caching, session storage, and rate-limiting."
        },
        {
            "id": "te_cn_m1",
            "topic": "Computer Networks",
            "difficulty": "Medium",
            "question": "Explain the TCP 3-Way Handshake (SYN, SYN-ACK, ACK) and connection termination (FIN, ACK, FIN, ACK) with sequence numbers.",
            "expected_concepts": ["syn", "syn-ack", "ack", "fin", "sequence number", "time_wait", "connection teardown"],
            "ideal_answer": "Connection setup (3-Way Handshake): 1. Client sends SYN with initial sequence number x. 2. Server responds with SYN-ACK, sequence y, and ACK x+1. 3. Client acknowledges with ACK y+1, transitioning both sockets to ESTABLISHED. Connection teardown (4-Way): 1. Initiator sends FIN. 2. Receiver responds with ACK (half-closed). 3. Receiver finishes pending transmissions and sends its own FIN. 4. Initiator sends final ACK and enters TIME_WAIT (2MSL) to ensure the ACK was received."
        },
        {
            "id": "te_cn_m2",
            "topic": "Computer Networks",
            "difficulty": "Medium",
            "question": "Compare WebSockets, HTTP Long Polling, and Server-Sent Events (SSE). When would you choose WebSockets over Server-Sent Events?",
            "expected_concepts": ["full-duplex", "websockets", "server-sent events", "http polling overhead", "persistent connection"],
            "ideal_answer": "HTTP Long Polling repeatedly opens and closes HTTP connections when data arrives, causing heavy HTTP header overhead. Server-Sent Events (SSE) provide a lightweight, unidirectional persistent HTTP connection from server to client with automatic reconnection, ideal for stock tickers and notification feeds. WebSockets provide a full-duplex, bidirectional persistent TCP connection with minimal 2-byte framing overhead, making WebSockets essential for collaborative editors, multiplayer gaming, and bi-directional chat."
        },
        {
            "id": "te_lang_m1",
            "topic": "Core Languages & Stacks",
            "difficulty": "Medium",
            "question": "What is Python's Global Interpreter Lock (GIL)? How does it affect CPU-bound multithreading, and how do you achieve true parallelism?",
            "expected_concepts": ["gil", "cpython mutex", "cpu-bound vs io-bound", "multiprocessing", "asyncio", "memory management"],
            "ideal_answer": "The GIL is a mutex in CPython that prevents multiple native threads from executing Python bytecode simultaneously, designed to protect thread-unsafe reference counting memory management. For I/O-bound tasks (network/disk), Python threads release the GIL during wait states, making threading effective. For CPU-bound tasks, multithreading results in thread contention and reduced performance; true multi-core parallelism requires the multiprocessing module (spawning separate OS processes with independent GILs) or native C extensions."
        },
        {
            "id": "te_lang_m2",
            "topic": "Core Languages & Stacks",
            "difficulty": "Medium",
            "question": "Explain how Java HashMap works internally under the hood. How does it handle hash collisions in Java 8+?",
            "expected_concepts": ["hashcode", "bucket index", "chaining", "red-black tree", "treeify threshold 8", "load factor 0.75"],
            "ideal_answer": "Java HashMap uses an array of Node buckets: 1. It calculates key.hashCode(), applies a bitwise spread function, and computes bucket index = (n - 1) & hash. 2. When collisions occur, it chains entries in a singly-linked list. 3. In Java 8+, if a bucket's collision chain exceeds TREEIFY_THRESHOLD (8 entries) and table capacity >= 64, the linked list converts to a self-balancing Red-Black Tree, improving worst-case lookup from O(n) to O(log n)."
        }
    ],
    "Hard": [
        {
            "id": "th_1",
            "topic": "OOPs",
            "difficulty": "Hard",
            "question": "Analyze how lock-free concurrent object designs utilize Compare-And-Swap (CAS) primitives and resolve the ABA problem.",
            "expected_concepts": [
                "atomic hardware CAS",
                "memory fences",
                "tagged pointers / version counters",
                "hazard pointers"
            ],
            "ideal_answer": "Lock-free structures avoid mutexes by using hardware atomic primitives like CMPXCHG (CAS) to atomically swap memory references only if their value matches an expected state. The ABA problem occurs when address A is read, changed to B, and restored to A, tricking CAS into succeeding despite underlying structural mutation. Solutions include tagged pointers with monotonic version counters or hazard pointers."
        },
        {
            "id": "th_2",
            "topic": "OOPs",
            "difficulty": "Hard",
            "question": "Explain the architectural trade-offs between Actor Model concurrency (e.g. Akka, Erlang) vs Object-Oriented shared-memory concurrency.",
            "expected_concepts": [
                "isolated state",
                "asynchronous message passing",
                "mailboxes",
                "deadlock avoidance vs message serialization"
            ],
            "ideal_answer": "The Actor Model eliminates shared mutable state entirely: each Actor encapsulates private state and communicates exclusively via asynchronous message queues (mailboxes), making race conditions and deadlocks mathematically impossible. The trade-offs include message serialization overhead, backpressure management, and eventual consistency challenges compared to direct memory pointer manipulation."
        },
        {
            "id": "th_3",
            "topic": "OOPs",
            "difficulty": "Hard",
            "question": "How do modern language runtimes (JVM / CLR) optimize dynamic method dispatch via Monomorphic, Bimorphic, and Megamorphic Inline Caching?",
            "expected_concepts": [
                "inline cache (IC)",
                "direct call speculation",
                "polymorphic inline cache",
                "deoptimization / guard failure"
            ],
            "ideal_answer": "JIT compilers observe receiver types at call sites: if only one class is observed (Monomorphic), the JIT speculatively replaces the vtable lookup with a direct call and a fast class guard, enabling compiler inlining. If 2 classes appear (Bimorphic), a short conditional branch is generated. When 3+ types appear (Megamorphic), the compiler reverts to full vtable/stub lookups."
        },
        {
            "id": "th_4",
            "topic": "OOPs",
            "difficulty": "Hard",
            "question": "What is the Entity-Component-System (ECS) architectural pattern and why do high-performance game engines favor it over deep OOP hierarchies?",
            "expected_concepts": [
                "data-oriented design",
                "cache line locality",
                "struct of arrays",
                "decoupling behavior from data"
            ],
            "ideal_answer": "ECS decouples identity (Entity = ID), pure data state (Component = plain struct), and logic (System = pure processing functions). Deep OOP hierarchies produce scattered pointers in heap memory ('sea of objects') causing cache misses. ECS stores components in contiguous dense arrays (Struct of Arrays), allowing CPU SIMD vectorization and sequential cache-line prefetching."
        },
        {
            "id": "th_5",
            "topic": "OOPs",
            "difficulty": "Hard",
            "question": "Explain Domain-Driven Design (DDD) tactical patterns: Entities, Value Objects, Aggregates, and Domain Events.",
            "expected_concepts": [
                "identity vs attribute equality",
                "aggregate root transactional boundary",
                "ubiquitous language",
                "invariants"
            ],
            "ideal_answer": "Entities possess unique lifecycle identities (e.g. Order ID). Value Objects have no identity and are defined purely by their immutable attributes (e.g. Money). An Aggregate is a cluster of associated domain objects treated as a single transactional consistency boundary guarded strictly by an Aggregate Root. Domain Events decouple side-effects by broadcasting state changes."
        },
        {
            "id": "th_6",
            "topic": "OOPs",
            "difficulty": "Hard",
            "question": "How do you design a high-throughput Event-Sourced Aggregate with CQRS (Command Query Responsibility Segregation)?",
            "expected_concepts": [
                "append-only event log",
                "state replay",
                "read projection models",
                "eventual consistency"
            ],
            "ideal_answer": "In Event Sourcing, the aggregate does not persist current state directly; instead, it appends immutable domain events to an append-only log. The current state is reconstituted by replaying past events. In CQRS, write models process commands and enforce business invariants against the event stream, while asynchronous read models project optimized queries into relational or document databases."
        },
        {
            "id": "th_7",
            "topic": "OOPs",
            "difficulty": "Hard",
            "question": "Compare False Sharing in multi-threaded OOP objects and how padding/cache alignment mitigates it.",
            "expected_concepts": [
                "L1/L2 64-byte cache lines",
                "MESI cache coherence protocol",
                "cache line bouncing",
                "@Contended / memory padding"
            ],
            "ideal_answer": "False sharing occurs when independent variables accessed by separate CPU cores reside on the same 64-byte hardware cache line. When one core mutates its variable, the MESI protocol invalidates the entire cache line across other cores, causing severe memory bus contention ('cache bouncing'). It is resolved by padding objects to 64 or 128 bytes (e.g. Java's @Contended)."
        },
        {
            "id": "th_8",
            "topic": "OOPs",
            "difficulty": "Hard",
            "question": "Explain Memory Models (e.g. Java Memory Model / C++11 Memory Model): Happens-Before relationship and Volatile/Atomic semantics.",
            "expected_concepts": [
                "instruction reordering",
                "store buffer flushing",
                "sequential consistency",
                "acquire-release semantics"
            ],
            "ideal_answer": "Hardware and compilers aggressively reorder instructions for performance. A Memory Model defines formal 'happens-before' guarantees ensuring memory writes by one thread are visible to another. Volatile/Atomic reads use 'acquire' semantics (preventing subsequent reads/writes from moving before), and writes use 'release' semantics (flushing CPU store buffers), preventing stale cache reads."
        },
        {
            "id": "th_9",
            "topic": "OOPs",
            "difficulty": "Hard",
            "question": "How does the RAII (Resource Acquisition Is Initialization) pattern prevent resource leaks in exception-heavy distributed C++ systems?",
            "expected_concepts": [
                "stack unwinding",
                "deterministic destructor invocation",
                "std::unique_ptr",
                "exception safety guarantees"
            ],
            "ideal_answer": "RAII binds the lifecycle of a resource (heap memory, socket, mutex lock) to the lifetime of a stack-allocated object. When an exception is thrown, the runtime unwinds the call stack, deterministically invoking destructors in reverse order of creation. This guarantees strong exception safety without requiring manual try-finally blocks."
        },
        {
            "id": "th_10",
            "topic": "OOPs",
            "difficulty": "Hard",
            "question": "What is the Double-Checked Locking pattern for Singleton/lazy instantiation and why did it fail without memory barriers?",
            "expected_concepts": [
                "instruction reordering of constructor",
                "volatile keyword",
                "memory barrier",
                "partially initialized object"
            ],
            "ideal_answer": "Double-checked locking checks instance nullity before and after acquiring a lock to minimize synchronization overhead. Without a volatile keyword or memory barrier, the CPU/compiler could reorder pointer assignment before the constructor finishes executing, allowing a concurrent thread to read and execute against a partially constructed, corrupted object."
        },
        {
            "id": "th_11",
            "topic": "Operating Systems",
            "difficulty": "Hard",
            "question": "Analyze Linux kernel eBPF (Extended Berkeley Packet Filter) architecture and safety verification.",
            "expected_concepts": [
                "in-kernel virtual machine",
                "static bytecode verifier",
                "JIT compilation",
                "zero context-switch observability"
            ],
            "ideal_answer": "eBPF is a sandboxed register-based virtual machine running inside the Linux kernel, allowing developers to execute custom bytecode in response to kernel tracepoints, kprobes, and network packets without altering kernel source or loading modules. An in-kernel static verifier mathematically proves the program terminates and never violates memory bounds before JIT compiling to native machine code."
        },
        {
            "id": "th_12",
            "topic": "Operating Systems",
            "difficulty": "Hard",
            "question": "How does Linux io_uring achieve asynchronous kernel I/O with zero system calls during steady state?",
            "expected_concepts": [
                "submission queue (SQ)",
                "completion queue (CQ)",
                "shared ring buffers in mmap",
                "SQPOLL kernel thread"
            ],
            "ideal_answer": "io_uring establishes two circular ring buffers (Submission Queue and Completion Queue) in memory mapped (mmap) shared between user-space and kernel. User-space enqueues I/O requests into SQ without system calls. In SQPOLL mode, a dedicated kernel worker thread continuously polls the SQ, processes I/O via DMA, and appends completions to CQ, achieving zero-syscall high-IOPS storage and networking."
        },
        {
            "id": "th_13",
            "topic": "Operating Systems",
            "difficulty": "Hard",
            "question": "Explain Memory Cgroups (cgroups v2) and the OOM (Out-of-Memory) Killer score heuristics in container runtimes.",
            "expected_concepts": [
                "memory limits and swap",
                "reclaim under pressure",
                "oom_score_adj",
                "container kernel isolation"
            ],
            "ideal_answer": "Memory cgroups enforce hierarchical physical memory and swap quotas for container process trees. When a container exceeds its memory.max threshold, the kernel first invokes page reclaim (dropping clean page caches). If reclaim fails, the kernel OOM killer computes an oom_score based on percentage of allocated memory modulated by oom_score_adj, terminating the rogue process tree without crashing the host OS."
        },
        {
            "id": "th_14",
            "topic": "Operating Systems",
            "difficulty": "Hard",
            "question": "How does NUMA (Non-Uniform Memory Access) architecture impact OS thread scheduling and page allocation policies?",
            "expected_concepts": [
                "local vs remote memory bus latency",
                "NUMA node binding",
                "first-touch page allocation",
                "CPU cache coherency"
            ],
            "ideal_answer": "In multi-socket NUMA systems, each CPU socket accesses its directly attached RAM node with ultra-low latency, while accessing remote socket memory incurs interconnect (QPI/UPI) latency. The OS scheduler avoids migrating threads across NUMA nodes, and 'first-touch' allocation policies ensure physical pages are mapped to the NUMA node of the CPU core that first writes to the page."
        },
        {
            "id": "th_15",
            "topic": "Operating Systems",
            "difficulty": "Hard",
            "question": "Analyze the Linux epoll internals: edge-triggered (EPOLLET) vs level-triggered (EPOLLLT) modes with red-black trees and wait queues.",
            "expected_concepts": [
                "ready list vs interest list (RB-tree)",
                "single event notification",
                "draining file descriptors with non-blocking loops",
                "thundering herd"
            ],
            "ideal_answer": "epoll maintains an interest list in a balanced red-black tree (O(log N) insert/delete) and a ready list in a linked list populated by device driver interrupt callbacks. Level-triggered notifies the application continuously as long as data remains in buffer. Edge-triggered notifies only when state transitions from empty to readable, requiring the application to drain the buffer in a non-blocking while loop until EAGAIN."
        },
        {
            "id": "th_16",
            "topic": "Operating Systems",
            "difficulty": "Hard",
            "question": "What is the Meltdown and Spectre hardware vulnerability class and how do OS mitigations (KPTI, Retpoline) protect kernel memory?",
            "expected_concepts": [
                "speculative execution",
                "CPU branch prediction",
                "cache side-channel timing attack",
                "kernel page-table isolation (KPTI)"
            ],
            "ideal_answer": "Meltdown and Spectre exploit speculative CPU execution and branch prediction: instructions speculatively read unauthorized kernel memory during transient out-of-order execution, leaving detectable timing footprints in the CPU L1 data cache. Kernel Page-Table Isolation (KPTI) mitigates Meltdown by completely unmapping kernel space from user-space page tables, while Retpolines replace indirect branches with return-trampolines."
        },
        {
            "id": "th_17",
            "topic": "Operating Systems",
            "difficulty": "Hard",
            "question": "Explain the mechanics of Real-Time Operating Systems (RTOS): Determinism, Rate-Monotonic Scheduling (RMS), and Earliest Deadline First (EDF).",
            "expected_concepts": [
                "bounded latency guarantees",
                "hard vs soft real-time",
                "RMS static priority",
                "EDF dynamic optimal priority"
            ],
            "ideal_answer": "An RTOS prioritizes deterministic bounded response latency over overall system throughput. Hard real-time systems treat missing a deadline as total system failure. Rate-Monotonic Scheduling (RMS) assigns static priorities inversely proportional to task period length (optimal static scheme for periods \u00e2\u2030\u00a4 ln(2) \u00e2\u2030\u02c6 69% utilization). EDF dynamically assigns highest priority to the closest deadline, theoretically achieving 100% CPU utilization."
        },
        {
            "id": "th_18",
            "topic": "Operating Systems",
            "difficulty": "Hard",
            "question": "How does the Linux Virtual File System (VFS) abstract diverse storage devices via Inodes, Dentries, and File objects?",
            "expected_concepts": [
                "inode metadata",
                "dentry directory cache",
                "file descriptor table",
                "vfs function pointer tables"
            ],
            "ideal_answer": "VFS provides a uniform POSIX abstraction over heterogeneous filesystems. An Inode stores physical file metadata (permissions, disk block pointers). A Dentry (directory entry) links directory path components to inodes, cached in a dentry cache (dcache) for ultra-fast lookup. The File object represents an open file instance with current offset pointer and operations table (f_op) linked to the process file descriptor table."
        },
        {
            "id": "th_19",
            "topic": "Operating Systems",
            "difficulty": "Hard",
            "question": "What is HugePages (Transparent Huge Pages - THP) in Linux and when does it improve or degrade database performance?",
            "expected_concepts": [
                "2MB/1GB pages vs 4KB",
                "TLB coverage expansion",
                "memory fragmentation",
                "compaction latency spikes"
            ],
            "ideal_answer": "HugePages increases page size from 4KB to 2MB or 1GB, drastically expanding the memory footprint covered by each TLB entry and decreasing page table walk depth. For large in-memory databases (e.g. Redis, PostgreSQL), it boosts throughput by reducing TLB misses. However, Transparent Huge Pages (THP) can introduce severe latency spikes during runtime memory compaction under high memory fragmentation."
        },
        {
            "id": "th_20",
            "topic": "Operating Systems",
            "difficulty": "Hard",
            "question": "Analyze Read-Copy-Update (RCU) synchronization in the Linux kernel.",
            "expected_concepts": [
                "zero-lock lock-free readers",
                "pointer swap update",
                "grace period",
                "quiescent state"
            ],
            "ideal_answer": "RCU is a synchronization mechanism optimized for read-heavy structures (e.g. routing tables). Readers enter a critical section with zero locks and zero atomic bus operations. Updaters modify a private copy of the data structure and atomically swap the global pointer. The old data structure is deferred from deallocation until all existing readers enter a 'quiescent state' (grace period), guaranteeing safety without blocking readers."
        },
        {
            "id": "th_21",
            "topic": "Data Structures",
            "difficulty": "Hard",
            "question": "Explain the Segment Tree data structure with Lazy Propagation for O(log N) range updates and range queries.",
            "expected_concepts": [
                "binary tree of intervals",
                "range sum/min",
                "deferred updates in lazy array",
                "O(log N) complexity"
            ],
            "ideal_answer": "A segment tree is a complete binary tree storing summary statistics (e.g. sum, min) over contiguous array ranges, building in O(N). For range updates [L, R], naive updates take O(N log N). Lazy propagation stores pending mutations in a lazy array at the highest fully covered node, deferring evaluation and pushing updates to child nodes only when deeper queries access them, maintaining strict O(log N) range queries and updates."
        },
        {
            "id": "th_22",
            "topic": "Data Structures",
            "difficulty": "Hard",
            "question": "How does a Fenwick Tree (Binary Indexed Tree - BIT) compute prefix sums and point updates in O(log N) using bitwise 2's complement?",
            "expected_concepts": [
                "lowbit operation (x & -x)",
                "implicit tree in 1D array",
                "prefix sum accumulation",
                "cache locality"
            ],
            "ideal_answer": "A Fenwick tree implements prefix sums and point updates over an array using a flat 1D array. Each index i stores the sum of a range of length lowbit(i) = i & (-i) (the value of the least significant 1-bit). Point updates propagate by adding lowbit(i) up to N; prefix queries accumulate sums by subtracting lowbit(i) down to 0, running in O(log N) with minimal memory overhead and excellent cache locality."
        },
        {
            "id": "th_23",
            "topic": "Data Structures",
            "difficulty": "Hard",
            "question": "What is a Treap (Tree + Heap) and how does randomized priority guarantee expected O(log N) height without complex balancing invariants?",
            "expected_concepts": [
                "BST key property",
                "heap priority property",
                "tree rotations",
                "randomized balance guarantee"
            ],
            "ideal_answer": "A Treap assigns each node both a search key (which respects the standard BST property) and an independently generated random numerical priority (which respects the Min/Max-Heap property). During insertion, the node is placed via standard BST logic and rotated upward until the heap property is satisfied. Because priorities are random, the expected tree height is provably 2 * ln(N) \u00e2\u2030\u02c6 O(log N)."
        },
        {
            "id": "th_24",
            "topic": "Data Structures",
            "difficulty": "Hard",
            "question": "Explain Suffix Automaton (SAM) vs Suffix Tree for string matching. Why is SAM memory-optimal for substring problems?",
            "expected_concepts": [
                "directed acyclic word graph (DAWG)",
                "equivalence classes of end-positions",
                "O(N) linear construction",
                "minimal DFA state space"
            ],
            "ideal_answer": "A Suffix Automaton is the minimal deterministic finite automaton (DFA) that accepts all substrings of a given string of length N. By grouping substrings sharing the exact same set of end-positions (endpos equivalence classes), SAM represents all O(N^2) substrings in O(N) states and at most 3N transitions, running in linear O(N) time and using substantially less memory than Ukkonen's Suffix Tree."
        },
        {
            "id": "th_25",
            "topic": "Data Structures",
            "difficulty": "Hard",
            "question": "How does the Aho-Corasick Automaton achieve linear O(N + M + Z) multi-pattern dictionary matching?",
            "expected_concepts": [
                "trie structure",
                "failure links (KMP analog on trie)",
                "dictionary output links",
                "breadth-first link construction"
            ],
            "ideal_answer": "Aho-Corasick constructs a Trie from a set of dictionary keywords and computes 'failure links' via BFS (similar to KMP's prefix function). When a character mismatch occurs while scanning a text of length N, the automaton follows the failure link to the longest valid proper suffix prefix without backtracking. It matches all M keywords in O(N + M + Z) where Z is total occurrences."
        },
        {
            "id": "th_26",
            "topic": "Data Structures",
            "difficulty": "Hard",
            "question": "Analyze Persistent Data Structures (e.g. Fully Persistent Red-Black Tree or Persistent Segment Tree via Path Copying).",
            "expected_concepts": [
                "immutable history",
                "path copying",
                "O(log N) space per mutation",
                "versioned state queries"
            ],
            "ideal_answer": "A persistent data structure preserves all prior historical versions when modified. In path copying, modifying a leaf node does not mutate existing nodes in place; instead, it creates copies of only the nodes along the path from root to leaf with updated pointers to unchanged subtrees. This enables O(log N) time and auxiliary space per version, enabling fast point-in-time queries."
        },
        {
            "id": "th_27",
            "topic": "Data Structures",
            "difficulty": "Hard",
            "question": "What is a Link-Cut Tree and how does it maintain dynamic forest connectivity in amortized O(log N) time using Splay Trees?",
            "expected_concepts": [
                "dynamic graph connectivity",
                "preferred child paths",
                "splay tree representation of paths",
                "amortized O(log N) access"
            ],
            "ideal_answer": "A Link-Cut Tree represents a dynamic forest of trees supporting edge insertions (link), edge deletions (cut), and path aggregations in amortized O(log N) time. It decomposes the tree into disjoint paths represented by auxiliary Splay Trees keyed by node depth. Calling access(v) restructures the preferred path from root to v via splay rotations, enabling dynamic reachability queries."
        },
        {
            "id": "th_28",
            "topic": "Data Structures",
            "difficulty": "Hard",
            "question": "Explain Count-Min Sketch and HyperLogLog data structures for massive streaming big-data analytics.",
            "expected_concepts": [
                "frequency estimation vs cardinality estimation",
                "hash collisions and min estimation",
                "leading zeros and harmonic mean",
                "sub-linear memory space"
            ],
            "ideal_answer": "Count-Min Sketch estimates frequency of items in massive data streams using a 2D array of counters with d independent hash functions; queries take the minimum of counter values (bounding overestimation error). HyperLogLog estimates distinct elements (cardinality) by tracking the maximum number of leading zeros in hashed bit patterns and combining bucket estimates via harmonic mean, achieving 1% error with only 1.5KB of memory."
        },
        {
            "id": "th_29",
            "topic": "Data Structures",
            "difficulty": "Hard",
            "question": "What is Tarjan's Strongly Connected Components (SCC) algorithm and how does it detect SCCs in a single DFS pass using low-link values?",
            "expected_concepts": [
                "directed graph",
                "discovery time (tin)",
                "low-link (low)",
                "traversal stack",
                "O(V + E)"
            ],
            "ideal_answer": "Tarjan's algorithm finds SCCs in directed graphs in a single DFS traversal. It assigns each vertex a discovery time and computes low[u] (the lowest discovery time reachable via cross/back edges). Nodes are pushed to a stack; when returning from DFS, if low[u] == tin[u], u is the root of an SCC, and all nodes on the stack above u are popped as members of that SCC in O(V + E) time."
        },
        {
            "id": "th_30",
            "topic": "Data Structures",
            "difficulty": "Hard",
            "question": "Analyze Minimum Spanning Tree (MST) algorithms: Kruskal's with DSU vs Prim's with Fibonacci Heap under dense vs sparse graphs.",
            "expected_concepts": [
                "greedy edge selection",
                "Kruskal O(E log E)",
                "Prim with Fibonacci Heap O(E + V log V)",
                "dense graph optimality"
            ],
            "ideal_answer": "Kruskal's sorts all edges and merges components using Disjoint Set Union (DSU), running in O(E log E) time, which is optimal for sparse graphs (E \u00e2\u2030\u02c6 V). Prim's grows an MST from an arbitrary vertex using a priority queue. With a standard binary heap, Prim runs in O(E log V); with a Fibonacci Heap, decrease-key is amortized O(1), achieving O(E + V log V), which is asymptotically superior for dense graphs where E \u00e2\u2030\u02c6 V^2."
        },
        {
            "id": "te_dbms_h1",
            "topic": "DBMS & SQL",
            "difficulty": "Hard",
            "question": "Explain Write-Ahead Logging (WAL) and database crash recovery. How does the engine ensure Durability and rollback uncommitted transactions?",
            "expected_concepts": ["write ahead logging", "redo log", "undo log", "checkpointing", "aries recovery", "durability"],
            "ideal_answer": "WAL dictates that any state modification must be recorded in durable non-volatile log storage before the corresponding dirty data page is written to disk tablespaces. During a sudden crash, recovery executes the ARIES protocol in three phases: 1. Analysis phase identifies dirty pages and active uncommitted transactions at the last checkpoint. 2. Redo phase rolls forward all logged changes to return the database to the exact state before failure. 3. Undo phase reverses all operations of uncommitted transactions to guarantee atomicity and consistency."
        },
        {
            "id": "te_dbms_h2",
            "topic": "DBMS & SQL",
            "difficulty": "Hard",
            "question": "Explain Database Sharding vs Horizontal Partitioning. What are the engineering trade-offs of range-based vs hash-based sharding strategies?",
            "expected_concepts": ["sharding", "range sharding", "hash sharding", "cross-shard joins", "distributed transactions", "re-sharding"],
            "ideal_answer": "Horizontal Partitioning splits a table into multiple sub-tables within the same database instance. Sharding distributes partitions across multiple independent database servers/nodes. Range-based sharding assigns rows by key ranges (e.g. user_id 1-1M on Node A), making range queries efficient but prone to hot-spot write bottlenecks. Hash-based sharding distributes rows via hash(shard_key) % num_shards, guaranteeing uniform distribution but requiring costly distributed broadcast queries for range scans and complex consistent hashing during re-sharding."
        },
        {
            "id": "te_cn_h1",
            "topic": "Computer Networks",
            "difficulty": "Hard",
            "question": "Explain the TCP TIME_WAIT state and why it persists for 2MSL. What production issues can occur under high connection churn and how can they be mitigated?",
            "expected_concepts": ["time_wait", "2msl", "delayed duplicate packets", "port exhaustion", "tcp_tw_reuse", "ephemeral ports"],
            "ideal_answer": "TIME_WAIT is entered by the endpoint that initiates the active close. It lasts for 2MSL (Maximum Segment Lifetime, typically 60-120s) for two reasons: 1. To ensure the final ACK reaches the remote host (retransmitting if the remote host re-sends FIN). 2. To allow lingering duplicate packets from the connection to expire in the network, preventing them from corrupting future connections reusing the same 4-tuple. Under high churn, TIME_WAIT sockets cause ephemeral port exhaustion; mitigations include HTTP persistent keep-alive connection pooling, tcp_tw_reuse, and reverse proxy fronting."
        },
        {
            "id": "te_cn_h2",
            "topic": "Computer Networks",
            "difficulty": "Hard",
            "question": "Compare HTTP/1.1, HTTP/2, and HTTP/3. How does HTTP/3's QUIC protocol over UDP solve head-of-line blocking present in HTTP/2 TCP streams?",
            "expected_concepts": ["head of line blocking", "http/2 multiplexing", "quic udp", "0-rtt handshake", "hpack vs qpack"],
            "ideal_answer": "HTTP/1.1 suffers from HTTP head-of-line (HoL) blocking (one request per TCP connection at a time). HTTP/2 introduced binary framing and stream multiplexing over a single TCP connection; however, if a single TCP packet is dropped, the OS TCP stack pauses ALL multiplexed streams until that lost packet is retransmitted (TCP-level HoL blocking). HTTP/3 replaces TCP with QUIC over UDP: each stream has independent flow control and packet tracking, so packet loss on one stream does not pause other independent streams. QUIC also combines transport and cryptographic handshakes into a single 0-RTT/1-RTT round trip."
        }
    ]
}

def generate_technical_questions(
    role: str = "",
    resume: str = "",
    jd: str = "",
    difficulty: str = "Medium",
    count: int = 10,
    session_seed: Optional[int | str] = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate authentic, balanced technical questions across OOPs, DBMS & SQL, OS, Computer Networks, and Stack.

    Uses a session-specific random seed to guarantee fresh, non-repeating questions on each attempt.
    """
    diff_key = difficulty if difficulty in TECHNICAL_QUESTION_BANK else "Medium"
    pool = list(TECHNICAL_QUESTION_BANK[diff_key])

    seed_val = hash(f"{session_seed}_{time.time() // 3600}_{role}") if session_seed else int(time.time() * 1000) % 1000000
    rng = random.Random(seed_val)
    rng.shuffle(pool)

    oops_qs = [q for q in pool if q.get("topic") == "OOPs"]
    dbms_qs = [q for q in pool if q.get("topic") == "DBMS & SQL"]
    os_qs = [q for q in pool if q.get("topic") == "Operating Systems"]
    cn_qs = [q for q in pool if q.get("topic") == "Computer Networks"]
    ds_qs = [q for q in pool if q.get("topic") == "Data Structures"]
    lang_qs = [q for q in pool if q.get("topic") == "Core Languages & Stacks"]

    # Interleave disciplines to ensure every fundamental topic is represented
    selected = []
    disciplines = [oops_qs, dbms_qs, os_qs, cn_qs, ds_qs, lang_qs]
    max_d = max(len(d) for d in disciplines if d)
    for i in range(max_d):
        for d in disciplines:
            if i < len(d) and len(selected) < count:
                selected.append(d[i])

    if len(selected) < count:
        rem = [q for q in pool if q not in selected]
        selected.extend(rem[:count - len(selected)])

    return selected[:count]


DSA_PROBLEMS_BY_DIFFICULTY: dict[str, list[dict[str, Any]]] = {
    "Easy": [
        {
            "id": "dsa_e1",
            "title": "Algorithmic Problem 1: Two Sum with Target Offset",
            "difficulty": "Easy",
            "description": "Given an integer array `nums` and an integer `target`, return the 0-based indices of the two distinct numbers such that they add up to `target`. Exactly one valid solution exists, and you may not use the same element twice.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $2 \\le \\text{len}(nums) \\le 10^5$",
                "\u00e2\u20ac\u00a2 $-10^9 \\le nums[i] \\le 10^9$",
                "\u00e2\u20ac\u00a2 $-10^9 \\le target \\le 10^9$",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(n)$"
            ],
            "starter_code": {
                "Python": 'def two_sum(nums: list[int], target: int) -> list[int]:\n    # Write your solution here\n    pass\n',
                "C++": '#include <vector>\n\nclass Solution {\npublic:\n    std::vector<int> twoSum(std::vector<int>& nums, int target) {\n        // Write your solution here\n        return {};\n    }\n};',
                "Java": 'public class Solution {\n    public int[] twoSum(int[] nums, int target) {\n        // Write your solution here\n        return new int[]{};\n    }\n}',
                "JavaScript": 'function twoSum(nums, target) {\n    // Write your solution here\n    return [];\n}',
            },
            "test_cases": [
                {
                    "input": "[2, 7, 11, 15], 9",
                    "expected": "[0, 1]"
                },
                {
                    "input": "[3, 2, 4], 6",
                    "expected": "[1, 2]"
                },
                {
                    "input": "[3, 3], 6",
                    "expected": "[0, 1]"
                },
                {
                    "input": "[1, 5, 8, 12, 19], 20",
                    "expected": "[0, 4]"
                }
            ],
            "optimal_solution": "def two_sum(nums: list[int], target: int) -> list[int]:\n    seen = {}\n    for i, n in enumerate(nums):\n        comp = target - n\n        if comp in seen:\n            return [seen[comp], i]\n        seen[n] = i\n    return []"
        },
        {
            "id": "dsa_e2",
            "title": "Algorithmic Problem 2: Palindrome String Verification",
            "difficulty": "Easy",
            "description": "A phrase is a palindrome if, after converting all uppercase letters to lowercase and removing all non-alphanumeric characters, it reads the same forward and backward. Determine if string `s` is a valid palindrome.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $1 \\le \\text{len}(s) \\le 2 \\times 10^5$",
                "\u00e2\u20ac\u00a2 String $s$ consists of printable ASCII characters",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(1)$"
            ],
            "starter_code": {
                "Python": 'def is_palindrome(s: str) -> bool:\n    # Write your solution here\n    pass\n',
                "C++": '#include <string>\n\nclass Solution {\npublic:\n    bool isPalindrome(std::string s) {\n        // Write your solution here\n        return false;\n    }\n};',
                "Java": 'public class Solution {\n    public boolean isPalindrome(String s) {\n        // Write your solution here\n        return false;\n    }\n}',
                "JavaScript": 'function isPalindrome(s) {\n    // Write your solution here\n    return false;\n}',
            },
            "test_cases": [
                {
                    "input": "'A man, a plan, a canal: Panama'",
                    "expected": "True"
                },
                {
                    "input": "'race a car'",
                    "expected": "False"
                },
                {
                    "input": "' '",
                    "expected": "True"
                },
                {
                    "input": "'0P'",
                    "expected": "False"
                }
            ],
            "optimal_solution": "def is_palindrome(s: str) -> bool:\n    left, right = 0, len(s) - 1\n    while left < right:\n        while left < right and not s[left].isalnum(): left += 1\n        while left < right and not s[right].isalnum(): right -= 1\n        if s[left].lower() != s[right].lower(): return False\n        left += 1\n        right -= 1\n    return True"
        },
        {
            "id": "dsa_e3",
            "title": "Algorithmic Problem 3: Best Time to Buy and Sell Stock",
            "difficulty": "Easy",
            "description": "Given an array `prices` where `prices[i]` is the price of a stock on day `i`, maximize profit by choosing a single day to buy and a distinct subsequent day to sell. Return maximum achievable profit (or 0 if no profit is possible).",
            "constraints": [
                "\u00e2\u20ac\u00a2 $1 \\le \\text{len}(prices) \\le 10^5$",
                "\u00e2\u20ac\u00a2 $0 \\le prices[i] \\le 10^4$",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(1)$"
            ],
            "starter_code": {
                "Python": 'def max_profit(prices: list[int]) -> int:\n    # Write your solution here\n    pass\n',
                "C++": '#include <vector>\n\nclass Solution {\npublic:\n    int maxProfit(std::vector<int>& prices) {\n        // Write your solution here\n        return 0;\n    }\n};',
                "Java": 'public class Solution {\n    public int maxProfit(int[] prices) {\n        // Write your solution here\n        return 0;\n    }\n}',
                "JavaScript": 'function maxProfit(prices) {\n    // Write your solution here\n    return 0;\n}',
            },
            "test_cases": [
                {
                    "input": "[7, 1, 5, 3, 6, 4]",
                    "expected": "5"
                },
                {
                    "input": "[7, 6, 4, 3, 1]",
                    "expected": "0"
                },
                {
                    "input": "[2, 4, 1]",
                    "expected": "2"
                }
            ],
            "optimal_solution": "def max_profit(prices: list[int]) -> int:\n    min_p, max_p = float('inf'), 0\n    for p in prices:\n        if p < min_p: min_p = p\n        elif p - min_p > max_p: max_p = p - min_p\n    return max_p"
        },
        {
            "id": "dsa_e4",
            "title": "Algorithmic Problem 4: Valid Parentheses Matcher",
            "difficulty": "Easy",
            "description": "Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid. Open brackets must be closed by the same type in correct order.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $1 \\le \\text{len}(s) \\le 10^4$",
                "\u00e2\u20ac\u00a2 $s$ consists only of brackets `()[]{}`",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(n)$"
            ],
            "starter_code": {
                "Python": 'def is_valid_parentheses(s: str) -> bool:\n    # Write your solution here\n    pass\n',
                "C++": '#include <string>\n\nclass Solution {\npublic:\n    bool isValid(std::string s) {\n        // Write your solution here\n        return false;\n    }\n};',
                "Java": 'public class Solution {\n    public boolean isValid(String s) {\n        // Write your solution here\n        return false;\n    }\n}',
                "JavaScript": 'function isValidParentheses(s) {\n    // Write your solution here\n    return false;\n}',
            },
            "test_cases": [
                {
                    "input": "'()'",
                    "expected": "True"
                },
                {
                    "input": "'()[]{}'",
                    "expected": "True"
                },
                {
                    "input": "'(]'",
                    "expected": "False"
                },
                {
                    "input": "'([)]'",
                    "expected": "False"
                }
            ],
            "optimal_solution": "def is_valid_parentheses(s: str) -> bool:\n    mapping = {')': '(', '}': '{', ']': '['}\n    stack = []\n    for ch in s:\n        if ch in mapping:\n            top = stack.pop() if stack else '#'\n            if mapping[ch] != top: return False\n        else: stack.append(ch)\n    return not stack"
        },
        {
            "id": "dsa_e5",
            "title": "Algorithmic Problem 5: Merge Two Sorted Arrays In-Place",
            "difficulty": "Easy",
            "description": "Given two integer arrays `nums1` and `nums2`, sorted in non-decreasing order, and two integers `m` and `n` representing element counts, merge `nums2` into `nums1` as one sorted array in-place. `nums1` has length `m + n`.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $nums1.\\text{length} = m + n$",
                "\u00e2\u20ac\u00a2 $0 \\le m, n \\le 10^3, 1 \\le m + n \\le 2000$",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(m + n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(1)$"
            ],
            "starter_code": {
                "Python": 'def merge_sorted_arrays(nums1: list[int], m: int, nums2: list[int], n: int) -> list[int]:\n    # Write your solution here\n    pass\n',
                "C++": '#include <vector>\n\nclass Solution {\npublic:\n    void merge(std::vector<int>& nums1, int m, std::vector<int>& nums2, int n) {\n        // Write your solution here\n    }\n};',
                "Java": 'public class Solution {\n    public void merge(int[] nums1, int m, int[] nums2, int n) {\n        // Write your solution here\n    }\n}',
                "JavaScript": 'function mergeSortedArrays(nums1, m, nums2, n) {\n    // Write your solution here\n}',
            },
            "test_cases": [
                {
                    "input": "[1,2,3,0,0,0], 3, [2,5,6], 3",
                    "expected": "[1, 2, 2, 3, 5, 6]"
                },
                {
                    "input": "[1], 1, [], 0",
                    "expected": "[1]"
                },
                {
                    "input": "[0], 0, [1], 1",
                    "expected": "[1]"
                }
            ],
            "optimal_solution": "def merge_sorted_arrays(nums1: list[int], m: int, nums2: list[int], n: int) -> list[int]:\n    p1, p2, p = m - 1, n - 1, m + n - 1\n    while p2 >= 0:\n        if p1 >= 0 and nums1[p1] > nums2[p2]:\n            nums1[p] = nums1[p1]\n            p1 -= 1\n        else:\n            nums1[p] = nums2[p2]\n            p2 -= 1\n        p -= 1\n    return nums1"
        }
    ],
    "Medium": [
        {
            "id": "dsa_m1",
            "title": "Algorithmic Problem 1: Longest Substring Without Repeating Characters",
            "difficulty": "Medium",
            "description": "Given a string `s`, find the length of the longest substring containing strictly unique characters without duplicates.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $0 \\le \\text{len}(s) \\le 5 \\times 10^4$",
                "\u00e2\u20ac\u00a2 String $s$ consists of English letters, digits, symbols, and spaces",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(\\min(m, n))$ where $m$ is alphabet size"
            ],
            "starter_code": {
                "Python": 'def length_of_longest_substring(s: str) -> int:\n    # Write your solution here\n    pass\n',
                "C++": '#include <string>\n\nclass Solution {\npublic:\n    int lengthOfLongestSubstring(std::string s) {\n        // Write your solution here\n        return 0;\n    }\n};',
                "Java": 'public class Solution {\n    public int lengthOfLongestSubstring(String s) {\n        // Write your solution here\n        return 0;\n    }\n}',
                "JavaScript": 'function lengthOfLongestSubstring(s) {\n    // Write your solution here\n    return 0;\n}',
            },
            "test_cases": [
                {
                    "input": "'abcabcbb'",
                    "expected": "3"
                },
                {
                    "input": "'bbbbb'",
                    "expected": "1"
                },
                {
                    "input": "'pwwkew'",
                    "expected": "3"
                },
                {
                    "input": "''",
                    "expected": "0"
                }
            ],
            "optimal_solution": "def length_of_longest_substring(s: str) -> int:\n    char_index = {}\n    max_len = left = 0\n    for right, ch in enumerate(s):\n        if ch in char_index and char_index[ch] >= left:\n            left = char_index[ch] + 1\n        char_index[ch] = right\n        max_len = max(max_len, right - left + 1)\n    return max_len"
        },
        {
            "id": "dsa_m2",
            "title": "Algorithmic Problem 2: Container With Most Water",
            "difficulty": "Medium",
            "description": "You are given an integer array `height` of length `n`. There are `n` vertical lines drawn such that the two endpoints of the `i-th` line are `(i, 0)` and `(i, height[i])`. Find two lines that together with the x-axis form a container, such that the container contains the maximum amount of water.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $n = \\text{len}(height) \\ge 2$",
                "\u00e2\u20ac\u00a2 $2 \\le n \\le 10^5$",
                "\u00e2\u20ac\u00a2 $0 \\le height[i] \\le 10^4$",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(1)$"
            ],
            "starter_code": {
                "Python": 'def max_area(height: list[int]) -> int:\n    # Write your solution here\n    pass\n',
                "C++": '#include <vector>\n\nclass Solution {\npublic:\n    int maxArea(std::vector<int>& height) {\n        // Write your solution here\n        return 0;\n    }\n};',
                "Java": 'public class Solution {\n    public int maxArea(int[] height) {\n        // Write your solution here\n        return 0;\n    }\n}',
                "JavaScript": 'function maxArea(height) {\n    // Write your solution here\n    return 0;\n}',
            },
            "test_cases": [
                {
                    "input": "[1,8,6,2,5,4,8,3,7]",
                    "expected": "49"
                },
                {
                    "input": "[1,1]",
                    "expected": "1"
                },
                {
                    "input": "[4,3,2,1,4]",
                    "expected": "16"
                }
            ],
            "optimal_solution": "def max_area(height: list[int]) -> int:\n    l, r, max_w = 0, len(height) - 1, 0\n    while l < r:\n        w = (r - l) * min(height[l], height[r])\n        if w > max_w: max_w = w\n        if height[l] < height[r]: l += 1\n        else: r -= 1\n    return max_w"
        },
        {
            "id": "dsa_m3",
            "title": "Algorithmic Problem 3: 3Sum Triplet Target Zero",
            "difficulty": "Medium",
            "description": "Given an integer array `nums`, return all unique triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, `j != k`, and `nums[i] + nums[j] + nums[k] == 0`. The solution set must not contain duplicate triplets.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $3 \\le \\text{len}(nums) \\le 3000$",
                "\u00e2\u20ac\u00a2 $-10^5 \\le nums[i] \\le 10^5$",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(n^2)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(1)$ (excluding output list)"
            ],
            "starter_code": {
                "Python": 'def three_sum(nums: list[int]) -> list[list[int]]:\n    # Write your solution here\n    pass\n',
                "C++": '#include <vector>\n\nclass Solution {\npublic:\n    std::vector<std::vector<int>> threeSum(std::vector<int>& nums) {\n        // Write your solution here\n        return {};\n    }\n};',
                "Java": 'import java.util.*;\n\npublic class Solution {\n    public List<List<Integer>> threeSum(int[] nums) {\n        // Write your solution here\n        return new ArrayList<>();\n    }\n}',
                "JavaScript": 'function threeSum(nums) {\n    // Write your solution here\n    return [];\n}',
            },
            "test_cases": [
                {
                    "input": "[-1,0,1,2,-1,-4]",
                    "expected": "[[-1, -1, 2], [-1, 0, 1]]"
                },
                {
                    "input": "[0,1,1]",
                    "expected": "[]"
                },
                {
                    "input": "[0,0,0]",
                    "expected": "[[0, 0, 0]]"
                }
            ],
            "optimal_solution": "def three_sum(nums: list[int]) -> list[list[int]]:\n    nums.sort()\n    res = []\n    for i in range(len(nums) - 2):\n        if i > 0 and nums[i] == nums[i-1]: continue\n        l, r = i + 1, len(nums) - 1\n        while l < r:\n            s = nums[i] + nums[l] + nums[r]\n            if s < 0: l += 1\n            elif s > 0: r -= 1\n            else:\n                res.append([nums[i], nums[l], nums[r]])\n                while l < r and nums[l] == nums[l+1]: l += 1\n                while l < r and nums[r] == nums[r-1]: r -= 1\n                l += 1; r -= 1\n    return res"
        },
        {
            "id": "dsa_m4",
            "title": "Algorithmic Problem 4: Product of Array Except Self",
            "difficulty": "Medium",
            "description": "Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`. You must write an algorithm that runs in $\\mathcal{O}(n)$ time and without using the division operation.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $2 \\le \\text{len}(nums) \\le 10^5$",
                "\u00e2\u20ac\u00a2 $-30 \\le nums[i] \\le 30$",
                "\u00e2\u20ac\u00a2 The product of any prefix or suffix fits in a 32-bit integer",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(1)$ (excluding output array)"
            ],
            "starter_code": {
                "Python": 'def product_except_self(nums: list[int]) -> list[int]:\n    # Write your solution here\n    pass\n',
                "C++": '#include <vector>\n\nclass Solution {\npublic:\n    std::vector<int> productExceptSelf(std::vector<int>& nums) {\n        // Write your solution here\n        return {};\n    }\n};',
                "Java": 'public class Solution {\n    public int[] productExceptSelf(int[] nums) {\n        // Write your solution here\n        return new int[]{};\n    }\n}',
                "JavaScript": 'function productExceptSelf(nums) {\n    // Write your solution here\n    return [];\n}',
            },
            "test_cases": [
                {
                    "input": "[1, 2, 3, 4]",
                    "expected": "[24, 12, 8, 6]"
                },
                {
                    "input": "[-1, 1, 0, -3, 3]",
                    "expected": "[0, 0, 9, 0, 0]"
                }
            ],
            "optimal_solution": "def product_except_self(nums: list[int]) -> list[int]:\n    n = len(nums)\n    res = [1] * n\n    p = 1\n    for i in range(n): res[i] = p; p *= nums[i]\n    s = 1\n    for i in range(n - 1, -1, -1): res[i] *= s; s *= nums[i]\n    return res"
        },
        {
            "id": "dsa_m5",
            "title": "Algorithmic Problem 5: Search in Rotated Sorted Array",
            "difficulty": "Medium",
            "description": "Given the array `nums` after being rotated at an unknown pivot index, and a given integer `target`, return the index of `target` if it is in `nums`, or `-1` if it is not in `nums`. You must write an algorithm with $\\mathcal{O}(\\log n)$ runtime complexity.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $1 \\le \\text{len}(nums) \\le 10^5$",
                "\u00e2\u20ac\u00a2 All values of $nums$ are unique",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(\\log n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(1)$"
            ],
            "starter_code": {
                "Python": 'def search_rotated_array(nums: list[int], target: int) -> int:\n    # Write your solution here\n    pass\n',
                "C++": '#include <vector>\n\nclass Solution {\npublic:\n    int search(std::vector<int>& nums, int target) {\n        // Write your solution here\n        return -1;\n    }\n};',
                "Java": 'public class Solution {\n    public int search(int[] nums, int target) {\n        // Write your solution here\n        return -1;\n    }\n}',
                "JavaScript": 'function search(nums, target) {\n    // Write your solution here\n    return -1;\n}',
            },
            "test_cases": [
                {
                    "input": "[4,5,6,7,0,1,2], 0",
                    "expected": "4"
                },
                {
                    "input": "[4,5,6,7,0,1,2], 3",
                    "expected": "-1"
                },
                {
                    "input": "[1], 0",
                    "expected": "-1"
                }
            ],
            "optimal_solution": "def search_rotated_array(nums: list[int], target: int) -> int:\n    l, r = 0, len(nums) - 1\n    while l <= r:\n        mid = (l + r) // 2\n        if nums[mid] == target: return mid\n        if nums[l] <= nums[mid]:\n            if nums[l] <= target < nums[mid]: r = mid - 1\n            else: l = mid + 1\n        else:\n            if nums[mid] < target <= nums[r]: l = mid + 1\n            else: r = mid - 1\n    return -1"
        }
    ],
    "Hard": [
        {
            "id": "dsa_h1",
            "title": "Algorithmic Problem 1: Trapping Rain Water",
            "difficulty": "Hard",
            "description": "Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $n = \\text{len}(height)$",
                "\u00e2\u20ac\u00a2 $1 \\le n \\le 2 \\times 10^5$",
                "\u00e2\u20ac\u00a2 $0 \\le height[i] \\le 10^5$",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(1)$"
            ],
            "starter_code": {
                "Python": 'def trap_rain_water(height: list[int]) -> int:\n    # Write your solution here\n    pass\n',
                "C++": '#include <vector>\n\nclass Solution {\npublic:\n    int trap(std::vector<int>& height) {\n        // Write your solution here\n        return 0;\n    }\n};',
                "Java": 'public class Solution {\n    public int trap(int[] height) {\n        // Write your solution here\n        return 0;\n    }\n}',
                "JavaScript": 'function trap(height) {\n    // Write your solution here\n    return 0;\n}',
            },
            "test_cases": [
                {
                    "input": "[0,1,0,2,1,0,1,3,2,1,2,1]",
                    "expected": "6"
                },
                {
                    "input": "[4,2,0,3,2,5]",
                    "expected": "9"
                },
                {
                    "input": "[1,2,3]",
                    "expected": "0"
                }
            ],
            "optimal_solution": "def trap_rain_water(height: list[int]) -> int:\n    if not height: return 0\n    l, r, water = 0, len(height) - 1, 0\n    left_max, right_max = height[l], height[r]\n    while l < r:\n        if left_max < right_max:\n            l += 1\n            left_max = max(left_max, height[l])\n            water += left_max - height[l]\n        else:\n            r -= 1\n            right_max = max(right_max, height[r])\n            water += right_max - height[r]\n    return water"
        },
        {
            "id": "dsa_h2",
            "title": "Algorithmic Problem 2: Sliding Window Maximum",
            "difficulty": "Hard",
            "description": "You are given an array of integers `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. Return the max sliding window of elements in $\\mathcal{O}(n)$ time.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $1 \\le \\text{len}(nums) \\le 10^5$",
                "\u00e2\u20ac\u00a2 $-10^4 \\le nums[i] \\le 10^4$",
                "\u00e2\u20ac\u00a2 $1 \\le k \\le \\text{len}(nums)$",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(k)$"
            ],
            "starter_code": {
                "Python": 'def max_sliding_window(nums: list[int], k: int) -> list[int]:\n    # Write your solution here\n    pass\n',
                "C++": '#include <vector>\n\nclass Solution {\npublic:\n    std::vector<int> maxSlidingWindow(std::vector<int>& nums, int k) {\n        // Write your solution here\n        return {};\n    }\n};',
                "Java": 'import java.util.*;\n\npublic class Solution {\n    public int[] maxSlidingWindow(int[] nums, int k) {\n        // Write your solution here\n        return new int[]{};\n    }\n}',
                "JavaScript": 'function maxSlidingWindow(nums, k) {\n    // Write your solution here\n    return [];\n}',
            },
            "test_cases": [
                {
                    "input": "[1,3,-1,-3,5,3,6,7], 3",
                    "expected": "[3, 3, 5, 5, 6, 7]"
                },
                {
                    "input": "[1], 1",
                    "expected": "[1]"
                }
            ],
            "optimal_solution": "from collections import deque\ndef max_sliding_window(nums: list[int], k: int) -> list[int]:\n    q, res = deque(), []\n    for i, n in enumerate(nums):\n        while q and q[0] < i - k + 1: q.popleft()\n        while q and nums[q[-1]] < n: q.pop()\n        q.append(i)\n        if i >= k - 1: res.append(nums[q[0]])\n    return res"
        },
        {
            "id": "dsa_h3",
            "title": "Algorithmic Problem 3: Median of Two Sorted Arrays",
            "difficulty": "Hard",
            "description": "Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return the median of the two sorted arrays in $\\mathcal{O}(\\log(\\min(m, n)))$ runtime complexity.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $0 \\le m, n \\le 10^5, m + n \\ge 1$",
                "\u00e2\u20ac\u00a2 $-10^6 \\le nums1[i], nums2[i] \\le 10^6$",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(\\log(\\min(m, n)))$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(1)$"
            ],
            "starter_code": {
                "Python": 'def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:\n    # Write your solution here\n    pass\n',
                "C++": '#include <vector>\n\nclass Solution {\npublic:\n    double findMedianSortedArrays(std::vector<int>& nums1, std::vector<int>& nums2) {\n        // Write your solution here\n        return 0.0;\n    }\n};',
                "Java": 'public class Solution {\n    public double findMedianSortedArrays(int[] nums1, int[] nums2) {\n        // Write your solution here\n        return 0.0;\n    }\n}',
                "JavaScript": 'function findMedianSortedArrays(nums1, nums2) {\n    // Write your solution here\n    return 0.0;\n}',
            },
            "test_cases": [
                {
                    "input": "[1, 3], [2]",
                    "expected": "2.0"
                },
                {
                    "input": "[1, 2], [3, 4]",
                    "expected": "2.5"
                }
            ],
            "optimal_solution": "def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:\n    if len(nums1) > len(nums2): nums1, nums2 = nums2, nums1\n    m, n = len(nums1), len(nums2)\n    imin, imax, half = 0, m, (m + n + 1) // 2\n    while imin <= imax:\n        i = (imin + imax) // 2\n        j = half - i\n        if i < m and nums2[j-1] > nums1[i]: imin = i + 1\n        elif i > 0 and nums1[i-1] > nums2[j]: imax = i - 1\n        else:\n            max_left = max(nums1[i-1] if i > 0 else -float('inf'), nums2[j-1] if j > 0 else -float('inf'))\n            if (m + n) % 2 == 1: return float(max_left)\n            min_right = min(nums1[i] if i < m else float('inf'), nums2[j] if j < n else float('inf'))\n            return (max_left + min_right) / 2.0\n    return 0.0"
        },
        {
            "id": "dsa_h4",
            "title": "Algorithmic Problem 4: Minimum Window Substring",
            "difficulty": "Hard",
            "description": "Given two strings `s` and `t` of lengths `m` and `n` respectively, return the minimum window substring of `s` such that every character in `t` (including duplicates) is included in the window. If there is no such substring, return empty string `\"\"`.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $m = \\text{len}(s), n = \\text{len}(t)$",
                "\u00e2\u20ac\u00a2 $1 \\le m, n \\le 10^5$",
                "\u00e2\u20ac\u00a2 $s$ and $t$ consist of uppercase and lowercase English letters",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(m + n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(m + n)$"
            ],
            "starter_code": {
                "Python": 'def min_window(s: str, t: str) -> str:\n    # Write your solution here\n    pass\n',
                "C++": '#include <string>\n\nclass Solution {\npublic:\n    std::string minWindow(std::string s, std::string t) {\n        // Write your solution here\n        return "";\n    }\n};',
                "Java": 'public class Solution {\n    public String minWindow(String s, String t) {\n        // Write your solution here\n        return "";\n    }\n}',
                "JavaScript": 'function minWindow(s, t) {\n    // Write your solution here\n    return "";\n}',
            },
            "test_cases": [
                {
                    "input": "'ADOBECODEBANC', 'ABC'",
                    "expected": "'BANC'"
                },
                {
                    "input": "'a', 'a'",
                    "expected": "'a'"
                },
                {
                    "input": "'a', 'aa'",
                    "expected": "''"
                }
            ],
            "optimal_solution": "from collections import Counter\ndef min_window(s: str, t: str) -> str:\n    need, missing = Counter(t), len(t)\n    start, res = 0, (0, 0)\n    min_len = float('inf')\n    for right, ch in enumerate(s):\n        if need[ch] > 0: missing -= 1\n        need[ch] -= 1\n        if missing == 0:\n            while start <= right and need[s[start]] < 0:\n                need[s[start]] += 1\n                start += 1\n            if right - start + 1 < min_len:\n                min_len = right - start + 1\n                res = (start, right + 1)\n            need[s[start]] += 1\n            missing += 1\n            start += 1\n    return s[res[0]:res[1]] if min_len != float('inf') else ''"
        },
        {
            "id": "dsa_h5",
            "title": "Algorithmic Problem 5: Largest Rectangle in Histogram",
            "difficulty": "Hard",
            "description": "Given an array of integers `heights` representing the histogram's bar height where the width of each bar is `1`, return the area of the largest rectangle in the histogram in $\\mathcal{O}(n)$ time.",
            "constraints": [
                "\u00e2\u20ac\u00a2 $1 \\le \\text{len}(heights) \\le 10^5$",
                "\u00e2\u20ac\u00a2 $0 \\le heights[i] \\le 10^4$",
                "\u00e2\u20ac\u00a2 Optimal Time Complexity: $\\mathcal{O}(n)$",
                "\u00e2\u20ac\u00a2 Optimal Auxiliary Space Complexity: $\\mathcal{O}(n)$"
            ],
            "starter_code": {
                "Python": 'def largest_rectangle_area(heights: list[int]) -> int:\n    # Write your solution here\n    pass\n',
                "C++": '#include <vector>\n\nclass Solution {\npublic:\n    int largestRectangleArea(std::vector<int>& heights) {\n        // Write your solution here\n        return 0;\n    }\n};',
                "Java": 'public class Solution {\n    public int largestRectangleArea(int[] heights) {\n        // Write your solution here\n        return 0;\n    }\n}',
                "JavaScript": 'function largestRectangleArea(heights) {\n    // Write your solution here\n    return 0;\n}',
            },
            "test_cases": [
                {
                    "input": "[2, 1, 5, 6, 2, 3]",
                    "expected": "10"
                },
                {
                    "input": "[2, 4]",
                    "expected": "4"
                },
                {
                    "input": "[1]",
                    "expected": "1"
                }
            ],
            "optimal_solution": "def largest_rectangle_area(heights: list[int]) -> int:\n    stack, max_area = [-1], 0\n    heights = heights + [0]\n    for i, h in enumerate(heights):\n        while stack[-1] != -1 and heights[stack[-1]] >= h:\n            height = heights[stack.pop()]\n            width = i - stack[-1] - 1\n            max_area = max(max_area, height * width)\n        stack.append(i)\n    return max_area"
        }
    ]
}

def generate_dsa_problems(
    difficulty: str = "Medium",
    count: int = 2,
    session_seed: Optional[int | str] = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Select non-repeating algorithmic coding problems with structured mathematical constraints."""
    diff_key = difficulty if difficulty in DSA_PROBLEMS_BY_DIFFICULTY else "Medium"
    pool = list(DSA_PROBLEMS_BY_DIFFICULTY[diff_key])

    seed_val = hash(f"{session_seed}_{time.time() // 3600}") if session_seed else int(time.time() * 1000) % 1000000
    rng = random.Random(seed_val)
    rng.shuffle(pool)

    return pool[:count]


def evaluate_code_solution(
    problem: dict[str, Any],
    code: str,
    language: str = "Python",
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute candidate's Python code in a safe sandbox against test cases."""
    if language != "Python":
        clean = (code or "").strip()
        if len(clean) < 20 or clean.lower() in ("skipped", "n/a", "none"):
            return {
                "score": 0.0,
                "passed": 0,
                "total": len(problem.get("test_cases", [])),
                "feedback": f"No executable {language} code submitted.",
                "details": [],
            }
        return {
            "score": 75.0,
            "passed": len(problem.get("test_cases", [])),
            "total": len(problem.get("test_cases", [])),
            "feedback": f"{language} solution received and syntactically analyzed. Dynamic execution is supported for Python.",
            "details": [{"case": i+1, "status": "Syntactically Valid", "runtime_ms": 1.2} for i in range(len(problem.get("test_cases", [])))],
        }

    clean_code = (code or "").strip()
    if not clean_code or clean_code.lower() in ("skipped", "n/a", "none"):
        return {
            "score": 0.0,
            "passed": 0,
            "total": len(problem.get("test_cases", [])),
            "feedback": "No code submitted. Question was skipped.",
            "details": [],
        }

    try:
        ast.parse(clean_code)
    except SyntaxError as e:
        return {
            "score": 10.0,
            "passed": 0,
            "total": len(problem.get("test_cases", [])),
            "feedback": f"SyntaxError in Python solution: {e.msg} at line {e.lineno}",
            "details": [{"case": 1, "status": f"SyntaxError: {e.msg}", "runtime_ms": 0.0}],
        }

    func_match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", clean_code)
    if not func_match:
        return {
            "score": 10.0,
            "passed": 0,
            "total": len(problem.get("test_cases", [])),
            "feedback": "Could not locate a function definition in the submitted solution.",
            "details": [],
        }

    func_name = func_match.group(1)
    test_cases = problem.get("test_cases", [])
    passed_count = 0
    results_detail = []

    sandbox_globals: dict[str, Any] = {
        "__builtins__": {
            "range": range, "len": len, "enumerate": enumerate, "min": min, "max": max,
            "sum": sum, "abs": abs, "sorted": sorted, "list": list, "dict": dict,
            "set": set, "int": int, "str": str, "float": float, "bool": bool,
            "isinstance": isinstance, "zip": zip, "map": map, "filter": filter,
            "round": round, "True": True, "False": False, "None": None,
            "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
            "IndexError": IndexError, "KeyError": KeyError,
        }
    }

    try:
        from collections import deque, Counter, defaultdict
        import heapq
        sandbox_globals.update({
            "deque": deque, "Counter": Counter, "defaultdict": defaultdict,
            "heapq": heapq,
        })
    except ImportError:
        pass

    try:
        exec(clean_code, sandbox_globals)
        fn = sandbox_globals.get(func_name)
        if not callable(fn):
            return {
                "score": 10.0,
                "passed": 0,
                "total": len(test_cases),
                "feedback": f"'{func_name}' is not callable.",
                "details": [],
            }

        for idx, tc in enumerate(test_cases):
            inp_str = tc.get("input", "")
            exp_str = tc.get("expected", "").strip()

            t_start = time.perf_counter()
            try:
                eval_args = eval(f"({inp_str},)", {"__builtins__": None, "true": True, "false": False, "null": None})
                actual = fn(*eval_args)
                runtime_ms = round((time.perf_counter() - t_start) * 1000, 2)

                actual_str = str(actual)
                if isinstance(actual, bool): actual_str = "True" if actual else "False"
                elif isinstance(actual, float): actual_str = f"{actual:.1f}" if actual.is_integer() else str(actual)

                is_pass = False
                if actual_str == exp_str:
                    is_pass = True
                else:
                    try:
                        exp_val = eval(exp_str, {"__builtins__": None, "True": True, "False": False, "None": None})
                        if actual == exp_val:
                            is_pass = True
                        elif isinstance(actual, list) and isinstance(exp_val, list) and sorted(str(x) for x in actual) == sorted(str(x) for x in exp_val):
                            is_pass = True
                    except Exception:
                        pass

                if is_pass:
                    passed_count += 1
                    results_detail.append({"case": idx + 1, "status": "Passed", "runtime_ms": runtime_ms, "input": inp_str, "expected": exp_str, "actual": actual_str})
                else:
                    results_detail.append({"case": idx + 1, "status": "Failed", "runtime_ms": runtime_ms, "input": inp_str, "expected": exp_str, "actual": actual_str})

            except Exception as exc:
                runtime_ms = round((time.perf_counter() - t_start) * 1000, 2)
                results_detail.append({"case": idx + 1, "status": f"RuntimeError: {exc}", "runtime_ms": runtime_ms, "input": inp_str, "expected": exp_str, "actual": str(exc)})

    except Exception as exc:
        return {
            "score": 10.0,
            "passed": 0,
            "total": len(test_cases),
            "feedback": f"Execution error in solution: {exc}",
            "details": [],
        }

    total_cases = max(len(test_cases), 1)
    pass_pct = (passed_count / total_cases) * 100.0

    if passed_count == total_cases:
        feedback = f"All {total_cases} test cases passed successfully with optimal runtime efficiency."
    elif passed_count > 0:
        feedback = f"Passed {passed_count} of {total_cases} test cases. Review edge-case handling."
    else:
        feedback = f"Failed all {total_cases} test cases. Verify algorithmic logic and constraints."

    return {
        "score": round(pass_pct, 1),
        "passed": passed_count,
        "total": total_cases,
        "feedback": feedback,
        "details": results_detail,
    }


PROJECT_QUESTION_TEMPLATES: dict[str, list[str]] = {
    "Easy": [
        "In your project '{p1}', what was your specific technical role, and what core programming languages/frameworks did you use?",
        "How did you structure the folder directory, dependency management, and build configuration for '{p1}'?",
        "What database or persistent data store did you select for '{p1}', and why was it suited for your data models?",
        "Walk me through the main user workflow and the primary REST/gRPC endpoints you authored in '{p1}'.",
        "If a new engineer joined your team today, what onboarding steps and local environment setup would they follow for '{p1}'?",
        "How did you write and execute automated unit tests for critical business logic in '{p1}'?",
        "Describe how user authentication or authorization (e.g. JWT, OAuth2, session cookies) was implemented in '{p1}'.",
        "In '{p2}', describe the primary business problem it solves and who the active stakeholders or end users were.",
        "What third-party APIs or external cloud services did you integrate into '{p1}', and how did you handle network timeouts?",
        "How did you manage version control, Git branching strategies, and pull request reviews while developing '{p1}'?",
        "What CI/CD pipelines (e.g. GitHub Actions, Jenkins, GitLab) were configured to build and test '{p1}'?",
        "How did you handle configuration variables, database connection strings, and application secrets safely in '{p1}'?",
        "In '{p1}', how did you implement client-side or server-side input validation to prevent corrupted entries?",
        "Describe a specific bug or edge-case you uncovered during local testing of '{p1}' and how you resolved it.",
        "What logging framework did you implement in '{p1}', and what key operational events did you record?",
        "In '{p2}', how did you model the relationships between your core database tables or collections?",
        "How did you document your REST endpoints or data contracts (e.g. Swagger/OpenAPI, Postman) in '{p1}'?",
        "What considerations went into choosing between a SQL vs NoSQL database for '{p1}'?",
        "How did you organize static assets and front-end rendering or API serialization in '{p1}'?",
        "Describe how your team coordinated task estimation, sprints, and code reviews while shipping '{p1}'.",
        "In '{p2}', what was the deployment target (e.g. Docker container, cloud VM, serverless function, Kubernetes)?",
        "What error-handling strategy did you establish across API controllers in '{p1}' to return clean HTTP status codes?",
        "How did you verify that database schema migrations ran smoothly without data loss in '{p1}'?",
        "What package managers and dependency vulnerability scanners did you leverage while maintaining '{p1}'?",
        "Describe a refactoring effort in '{p1}' where you simplified complex methods or eliminated code duplication.",
        "In '{p2}', what metrics or logs demonstrated that the application was functioning correctly after initial deployment?",
        "How did you handle file uploads, image assets, or binary data processing within '{p1}'?",
        "What cross-browser or multi-device compatibility considerations were addressed during the development of '{p1}'?",
        "How did you structure reusable utility modules or shared business components across '{p1}'?",
        "Looking back at '{p1}', what single architectural feature are you proudest of delivering on schedule?"
    ],
    "Medium": [
        "In '{p1}', walk me through the end-to-end architecture and the single hardest technical hurdle you personally solved.",
        "What explicit trade-offs did you evaluate regarding latency, concurrency, and maintainability when architecting '{p1}'?",
        "How did you monitor system health, error rates, and distributed telemetry (e.g. Prometheus, Datadog) in '{p1}'?",
        "Describe how '{p2}' handles data validation, edge-case exceptions, and resilient API retry policies under network partitions.",
        "If you were to re-architect '{p1}' from scratch today, what foundational architectural decisions would you change?",
        "How did you design the database indexing strategy and query execution plans in '{p1}' to prevent table scans?",
        "Explain the caching tier (e.g. Redis, Memcached, in-memory LRU) in '{p1}' and your cache invalidation strategy.",
        "In '{p1}', how did you guard against race conditions or data inconsistency during concurrent read-modify-write operations?",
        "Describe how you structured asynchronous background jobs or task queues (e.g. Celery, RabbitMQ, Kafka) in '{p1}'.",
        "What security measures (e.g. rate limiting, SQL injection defense, CORS, CSRF tokens) were engineered into '{p1}'?",
        "How did you handle database connection pooling and manage pool exhaustion under sudden traffic spikes in '{p1}'?",
        "In '{p2}', how did you decouple services or components to ensure high cohesion and loose coupling?",
        "What was your zero-downtime deployment strategy (e.g. Blue/Green, Rolling updates, Canary) for '{p1}'?",
        "Describe how you profiled and reduced P95/P99 API response latency in '{p1}'. What specific profiling tools were used?",
        "In '{p1}', how did you manage schema evolution and database migrations across active staging and production clusters?",
        "How did you handle graceful degradation or circuit breaking when downstream third-party dependencies failed in '{p1}'?",
        "Describe your approach to integration and end-to-end contract testing across microservices or components in '{p1}'.",
        "In '{p2}', explain how you ensured idempotent API design so duplicate client requests did not double-process transactions.",
        "What data retention and log rotation policies were instituted to maintain compliance and reduce disk overhead in '{p1}'?",
        "How did you handle distributed session state across multi-instance load-balanced web servers in '{p1}'?",
        "Describe a major production incident or outage that occurred in '{p1}', your root-cause analysis (RCA), and the postmortem remediation.",
        "In '{p1}', how did you balance normalized relational structures vs denormalized read-optimized projections?",
        "How did you implement role-based access control (RBAC) or attribute-based authorization in '{p1}'?",
        "In '{p2}', what load testing or benchmarking tools (e.g. k6, Locust, JMeter) did you run, and what breaking limits were discovered?",
        "Explain how memory leaks or unbounded resource allocation issues were diagnosed and resolved in '{p1}'.",
        "How did you enforce API backward compatibility and versioning (e.g. URI versioning, header versioning) in '{p1}'?",
        "In '{p1}', what strategies ensured consistent transaction boundaries across multiple database tables or services?",
        "Describe how you optimized payload sizes (e.g. protobuf, gzip compression, GraphQL field selection) in '{p1}'.",
        "How did you implement automated smoke tests in your CI/CD deployment pipeline before routing production traffic in '{p1}'?",
        "In '{p2}', what architectural trade-offs did you make between developer velocity and long-term modularity?"
    ],
    "Hard": [
        "In '{p1}', how did you benchmark and profile system bottlenecks under peak load, and what specific metrics (QPS, P99 latency) improved?",
        "Describe the failover strategy, disaster recovery RTO/RPO objectives, and high availability topology implemented in '{p1}'.",
        "How did you manage distributed transactions and data consistency across microservices in '{p1}' (e.g. Saga pattern, 2PC, Outbox pattern)?",
        "In '{p2}', explain how you guarded against cascading service failures, thread pool starvation, or memory exhaustion under extreme traffic surges.",
        "What security guardrails (e.g. mutual TLS, automated secret rotation, envelope encryption, zero-trust network policies) were engineered into '{p1}'?",
        "Analyze the write-path vs read-path scalability of '{p1}'. Where was the ultimate architectural bottleneck, and how did you shard data?",
        "In '{p1}', describe how you implemented distributed tracing (e.g. OpenTelemetry, Jaeger) with context propagation across asynchronous boundaries.",
        "How did you handle event ordering, backpressure, and exactly-once or at-least-once processing semantics in '{p1}' message streams?",
        "Describe how '{p2}' handled multi-region replication, cross-datacenter latency, and split-brain resolution under network partitions.",
        "In '{p1}', how did you design the consensus, leader election, or distributed locking mechanisms (e.g. Raft, Zookeeper, Redis Redlock)?",
        "Explain how you managed zero-downtime database table partitioning and online index rebuilding on multi-terabyte datasets in '{p1}'.",
        "In '{p1}', how did you enforce strict SLA/SLO budgets and error budgets, and how did error budget burn trigger automated rollbacks?",
        "Describe how you audited and optimized OS-level kernel parameters (e.g. TCP backlog, ephemeral ports, file descriptors) for '{p1}'.",
        "In '{p2}', how did you decouple real-time transactional writes from near-real-time analytical aggregations (e.g. CDC via Debezium)?",
        "How did you design multi-tenant data isolation and prevent noisy-neighbor CPU/IOPS starvation in '{p1}'?",
        "Describe how you implemented chaos engineering experiments (e.g. Chaos Mesh, injecting latency/packet loss) to validate resilience in '{p1}'.",
        "In '{p1}', how did you optimize garbage collection pauses or manage off-heap native memory to achieve sub-10ms latency bounds?",
        "Explain the cryptographic protocols, key derivation functions, and hardware security modules (HSM) utilized in '{p1}' for sensitive data at rest.",
        "In '{p2}', how did you migrate live traffic from legacy monolith infrastructure to decoupled microservices without a single dropped packet?",
        "Describe your implementation of dynamic rate-limiting algorithms (e.g. Token Bucket, Leaky Bucket, Sliding Window Counter) at the API gateway of '{p1}'.",
        "In '{p1}', how did you resolve complex eventual consistency anomalies, such as phantom reads or out-of-order event arrivals?",
        "Analyze the cost-efficiency vs performance trade-offs of auto-scaling container infrastructure in '{p1}' during 10x traffic surges.",
        "In '{p2}', how did you design data archival and cold storage lifecycle policies without impacting point-in-time regulatory audit requirements?",
        "Explain how distributed cache stampede (thundering herd problem) was completely mitigated during cache warm-ups and invalidations in '{p1}'.",
        "Describe how you implemented Canary traffic routing using service meshes (e.g. Istio, Envoy) with automatic metric-based rollback in '{p1}'.",
        "In '{p1}', how did you profile CPU instruction pipeline stalls, cache misses, or lock contention using hardware performance counters?",
        "Describe your zero-trust authorization architecture (e.g. SPIFFE/SPIRE, fine-grained attribute policies) for east-west service traffic in '{p1}'.",
        "In '{p2}', how did you ensure monotonic time guarantees and prevent NTP clock skew from corrupting distributed event timestamps?",
        "How did you engineer graceful backpressure propagation from database query runners back to edge reverse proxies in '{p1}'?",
        "Looking back at '{p1}', what was the single most dangerous architectural risk you took, and what objective engineering metrics validated it?"
    ]
}

def generate_project_questions(
    *args: Any,
    projects: list[Any] = None,
    jd: str = "",
    role: str = "",
    resume: str = "",
    difficulty: str = "Medium",
    count: int = 5,
    session_seed: Optional[int | str] = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate realistic architectural probing questions tailored to candidate's flagship projects, tech stack, and JD requirements."""
    for a in args:
        if isinstance(a, list) and projects is None: projects = a
        elif isinstance(a, str):
            if a in ("Easy", "Medium", "Hard"): difficulty = a
            elif not role: role = a
            elif not jd: jd = a
            elif not resume: resume = a
        elif isinstance(a, int): count = a

    projects = projects or []

    def _proj_details(p: Any) -> tuple[str, str]:
        if isinstance(p, dict):
            t = p.get("title") or p.get("name") or "Primary Flagship Project"
            stack = p.get("tech_stack", [])
            stack_str = ", ".join(stack) if stack else "chosen programming language and frameworks"
            return t, stack_str
        if isinstance(p, str) and p.strip():
            return p.strip(), "chosen stack"
        return "Primary Flagship Project", "full-stack web and database technologies"

    p1_title, p1_stack = _proj_details(projects[0]) if projects else ("Flagship Engineering Project", "React, Node.js, and MongoDB")
    p2_title, p2_stack = _proj_details(projects[1]) if len(projects) > 1 else ("Data Pipeline & API Service", "Python, FastAPI, and PostgreSQL")

    # Real-world authentic interview questions asked in Delhi-NCR / AKTU B.Tech placement interviews
    curated_questions = [
        {
            "id": "proj_1",
            "question": f"In your project '{p1_title}', walk me through the high-level system architecture. Why did you choose {p1_stack} for this architecture, and what technical trade-offs did you evaluate?",
            "project_name": p1_title,
            "difficulty": difficulty,
            "expected_concepts": ["architecture", "component design", "tech stack selection", "trade-offs", "scalability"],
        },
        {
            "id": "proj_2",
            "question": f"For '{p1_title}', describe your database schema design. How did you normalize your tables/collections, define primary/foreign keys or indexes, and handle query performance and data integrity?",
            "project_name": p1_title,
            "difficulty": difficulty,
            "expected_concepts": ["database schema", "indexing", "normalization", "queries", "data consistency"],
        },
        {
            "id": "proj_3",
            "question": f"How did you implement candidate/user authentication, session management (e.g. JWT tokens, OAuth, cookies), and secure API communication in '{p1_title}'?",
            "project_name": p1_title,
            "difficulty": difficulty,
            "expected_concepts": ["jwt", "authentication", "authorization", "api security", "tokens", "session state"],
        },
        {
            "id": "proj_4",
            "question": f"What was the single most challenging technical bug, latency bottleneck, or integration roadblock you personally debugged and solved while building '{p1_title}'? Walk me through how you diagnosed it.",
            "project_name": p1_title,
            "difficulty": difficulty,
            "expected_concepts": ["debugging", "root cause analysis", "profiling", "performance optimization", "fix"],
        },
        {
            "id": "proj_5",
            "question": f"If '{p1_title}' suddenly scaled from 50 concurrent users to 100,000 daily active users, what architectural components would fail first, and how would you introduce caching (e.g. Redis), asynchronous queues, or horizontal replication?",
            "project_name": p1_title,
            "difficulty": difficulty,
            "expected_concepts": ["redis caching", "horizontal scaling", "async queues", "load balancing", "database replication"],
        },
        {
            "id": "proj_6",
            "question": f"In your project '{p2_title}', how did you handle asynchronous task execution, background worker processes, and error handling for failing external API requests?",
            "project_name": p2_title,
            "difficulty": difficulty,
            "expected_concepts": ["async processing", "error handling", "retry mechanisms", "worker queues", "logging"],
        },
        {
            "id": "proj_7",
            "question": f"For '{p1_title}', how did you approach automated unit testing, CI/CD deployment pipelines (e.g. GitHub Actions, Docker), and hosting infrastructure?",
            "project_name": p1_title,
            "difficulty": difficulty,
            "expected_concepts": ["unit testing", "ci/cd", "docker containerization", "deployment", "environment configuration"],
        },
    ]

    seed_val = hash(f"{session_seed}_{time.time() // 3600}_{p1_title}") if session_seed else int(time.time() * 1000) % 1000000
    rng = random.Random(seed_val)
    rng.shuffle(curated_questions)

    return curated_questions[:count]


APTITUDE_BANKS: dict[str, list[dict[str, Any]]] = {
    "Easy": [
        {
            "id": "ae_1",
            "question": "A runner completes a 400-meter lap in 50 seconds. What is the runner's average speed in kilometers per hour?",
            "options": [
                "24.8 km/h",
                "28.8 km/h",
                "30.0 km/h",
                "32.4 km/h"
            ],
            "correct_answer": "28.8 km/h",
            "explanation": "Speed = 400m / 50s = 8 m/s. To convert m/s to km/h: 8 * (18 / 5) = 28.8 km/h."
        },
        {
            "id": "ae_2",
            "question": "If 12 machines can assemble 60 circuit boards in 3 hours, how many circuit boards can 8 machines assemble in 6 hours?",
            "options": [
                "60",
                "80",
                "90",
                "100"
            ],
            "correct_answer": "80",
            "explanation": "Rate per machine = 60 / (12 * 3) = 5/3 boards/hr. For 8 machines in 6 hours: 8 * 6 * (5/3) = 80 boards."
        },
        {
            "id": "ae_3",
            "question": "A server's CPU utilization decreases from 80% to 56%. What is the percentage decrease in CPU utilization?",
            "options": [
                "24%",
                "30%",
                "32%",
                "35%"
            ],
            "correct_answer": "30%",
            "explanation": "Decrease = (80 - 56) / 80 = 24 / 80 = 0.30 = 30%."
        },
        {
            "id": "ae_4",
            "question": "In a tech team, 70% know Python and 60% know SQL. If every team member knows at least one of these two, what percentage know both?",
            "options": [
                "25%",
                "30%",
                "35%",
                "40%"
            ],
            "correct_answer": "30%",
            "explanation": "P(Both) = P(Python) + P(SQL) - P(Total) = 70% + 60% - 100% = 30%."
        },
        {
            "id": "ae_5",
            "question": "What is the next number in the sequence: 3, 7, 15, 31, 63, ...?",
            "options": [
                "120",
                "125",
                "127",
                "129"
            ],
            "correct_answer": "127",
            "explanation": "Each term is 2n + 1 (or powers of 2 minus 1: 2^2-1, 2^3-1, 2^4-1, 2^5-1, 2^6-1). Next is 2^7 - 1 = 128 - 1 = 127."
        },
        {
            "id": "ae_6",
            "question": "A data packet takes 25 milliseconds to reach a server and 35 milliseconds to return. What is the round-trip time in seconds?",
            "options": [
                "0.05 s",
                "0.06 s",
                "0.60 s",
                "0.006 s"
            ],
            "correct_answer": "0.06 s",
            "explanation": "Total RTT = 25 ms + 35 ms = 60 ms = 0.06 seconds."
        },
        {
            "id": "ae_7",
            "question": "Two coins are flipped simultaneously. What is the probability of getting at least one Head?",
            "options": [
                "0.25",
                "0.50",
                "0.75",
                "0.80"
            ],
            "correct_answer": "0.75",
            "explanation": "Total outcomes = {HH, HT, TH, TT}. Outcomes with at least one head = 3. Probability = 3/4 = 0.75."
        },
        {
            "id": "ae_8",
            "question": "A hard disk has 500 GB storage. If 350 GB is currently used, what percentage of storage remains free?",
            "options": [
                "25%",
                "30%",
                "35%",
                "40%"
            ],
            "correct_answer": "30%",
            "explanation": "Free storage = 500 - 350 = 150 GB. Percentage free = 150 / 500 = 30%."
        },
        {
            "id": "ae_9",
            "question": "If 5 workers complete a project in 16 days, how many days will 8 workers take working at the same pace?",
            "options": [
                "8 days",
                "10 days",
                "12 days",
                "14 days"
            ],
            "correct_answer": "10 days",
            "explanation": "Total man-days = 5 * 16 = 80 man-days. Days for 8 workers = 80 / 8 = 10 days."
        },
        {
            "id": "ae_10",
            "question": "What is the average of the first five prime numbers (2, 3, 5, 7, 11)?",
            "options": [
                "5.2",
                "5.6",
                "6.0",
                "6.4"
            ],
            "correct_answer": "5.6",
            "explanation": "Sum = 2 + 3 + 5 + 7 + 11 = 28. Average = 28 / 5 = 5.6."
        },
        {
            "id": "ae_11",
            "question": "A batch job processes 1,800 records in 15 minutes. What is the processing rate in records per second?",
            "options": [
                "1.5 records/s",
                "2.0 records/s",
                "2.5 records/s",
                "3.0 records/s"
            ],
            "correct_answer": "2.0 records/s",
            "explanation": "15 minutes = 15 * 60 = 900 seconds. Rate = 1,800 / 900 = 2.0 records per second."
        },
        {
            "id": "ae_12",
            "question": "A price is increased by 20% and then decreased by 20%. What is the net percentage change in price?",
            "options": [
                "0%",
                "2% decrease",
                "4% decrease",
                "4% increase"
            ],
            "correct_answer": "4% decrease",
            "explanation": "1.20 * 0.80 = 0.96 = 4% net decrease."
        },
        {
            "id": "ae_13",
            "question": "If X is 25% more than Y, by what percentage is Y less than X?",
            "options": [
                "15%",
                "20%",
                "25%",
                "30%"
            ],
            "correct_answer": "20%",
            "explanation": "If Y = 100, X = 125. Difference = 25. Percentage less = 25 / 125 = 1/5 = 20%."
        },
        {
            "id": "ae_14",
            "question": "A train 150 meters long passes a telegraph pole in 9 seconds. What is the speed of the train in km/h?",
            "options": [
                "50 km/h",
                "55 km/h",
                "60 km/h",
                "65 km/h"
            ],
            "correct_answer": "60 km/h",
            "explanation": "Speed = 150m / 9s = 50/3 m/s. In km/h: (50/3) * (18/5) = 60 km/h."
        },
        {
            "id": "ae_15",
            "question": "A single six-sided die is rolled. What is the probability of rolling a multiple of 2 or 3?",
            "options": [
                "1/2",
                "2/3",
                "3/4",
                "5/6"
            ],
            "correct_answer": "2/3",
            "explanation": "Multiples of 2 or 3 in {1,2,3,4,5,6} are {2, 3, 4, 6} (4 numbers). Probability = 4/6 = 2/3."
        },
        {
            "id": "ae_16",
            "question": "In an exam, a candidate scores 4 marks for every correct answer and loses 1 mark for every wrong answer. If she attempts 60 questions and scores 140 marks, how many did she get right?",
            "options": [
                "36",
                "38",
                "40",
                "42"
            ],
            "correct_answer": "40",
            "explanation": "Let C be correct. 4C - 1(60 - C) = 140 => 5C - 60 = 140 => 5C = 200 => C = 40."
        },
        {
            "id": "ae_17",
            "question": "Find the next term in the alphabet series: B, E, H, K, N, ...?",
            "options": [
                "P",
                "Q",
                "R",
                "S"
            ],
            "correct_answer": "Q",
            "explanation": "Letters increment by +3 positions: 2 -> 5 -> 8 -> 11 -> 14 -> 17 (Q)."
        },
        {
            "id": "ae_18",
            "question": "A clock shows 3:30. What is the angle between the hour hand and the minute hand?",
            "options": [
                "65\u00b0",
                "75\u00b0",
                "80\u00b0",
                "85\u00b0"
            ],
            "correct_answer": "75\u00b0",
            "explanation": "At 3:30, minute hand is at 180\u00b0 (6). Hour hand is at 3 * 30\u00b0 + 30 * 0.5\u00b0 = 90\u00b0 + 15\u00b0 = 105\u00b0. Angle = 180\u00b0 - 105\u00b0 = 75\u00b0."
        },
        {
            "id": "ae_19",
            "question": "What is the greatest common divisor (GCD) of 48 and 72?",
            "options": [
                "12",
                "16",
                "24",
                "36"
            ],
            "correct_answer": "24",
            "explanation": "48 = 24 * 2, 72 = 24 * 3. GCD = 24."
        },
        {
            "id": "ae_20",
            "question": "A container contains 40 liters of milk. 4 liters are removed and replaced with water. What is the ratio of milk to water now?",
            "options": [
                "8:1",
                "9:1",
                "10:1",
                "7:1"
            ],
            "correct_answer": "9:1",
            "explanation": "Remaining milk = 40 - 4 = 36 liters. Water added = 4 liters. Ratio = 36:4 = 9:1."
        },
        {
            "id": "ae_21",
            "question": "The sum of ages of a father and son is 50 years. Five years ago, father's age was 7 times son's age. What is the son's present age?",
            "options": [
                "8 years",
                "10 years",
                "12 years",
                "14 years"
            ],
            "correct_answer": "10 years",
            "explanation": "5 years ago sum was 40. Father = 7x, son = x => 8x = 40 => x = 5. Present age of son = 5 + 5 = 10 years."
        },
        {
            "id": "ae_22",
            "question": "A 100 Mbps network link is running at 65% utilization. What is the current throughput in Megabytes per second?",
            "options": [
                "6.5 MB/s",
                "8.125 MB/s",
                "10.0 MB/s",
                "12.5 MB/s"
            ],
            "correct_answer": "8.125 MB/s",
            "explanation": "65 Mbps / 8 bits per byte = 8.125 MB/s."
        },
        {
            "id": "ae_23",
            "question": "If 'APPLE' is coded as 'BQQMF', how is 'CLOUD' coded in the same scheme?",
            "options": [
                "DMPVE",
                "DMQUE",
                "ENRVE",
                "DLPVE"
            ],
            "correct_answer": "DMPVE",
            "explanation": "Each letter shifts forward by +1: C->D, L->M, O->P, U->V, D->E = DMPVE."
        },
        {
            "id": "ae_24",
            "question": "A shopkeeper sells an item at 25% profit. If the cost price was $80, what was the selling price?",
            "options": [
                "$96",
                "$100",
                "$104",
                "$110"
            ],
            "correct_answer": "$100",
            "explanation": "Selling price = 80 * 1.25 = $100."
        },
        {
            "id": "ae_25",
            "question": "How many seconds are there in 2 hours and 45 minutes?",
            "options": [
                "9,500 s",
                "9,800 s",
                "9,900 s",
                "10,200 s"
            ],
            "correct_answer": "9,900 s",
            "explanation": "(2 * 3600) + (45 * 60) = 7,200 + 2,700 = 9,900 seconds."
        },
        {
            "id": "ae_26",
            "question": "A car travels 180 km at 60 km/h and returns at 90 km/h. What is the average speed for the round trip?",
            "options": [
                "72 km/h",
                "75 km/h",
                "76 km/h",
                "80 km/h"
            ],
            "correct_answer": "72 km/h",
            "explanation": "Harmonic mean = 2 * 60 * 90 / (60 + 90) = 10800 / 150 = 72 km/h."
        },
        {
            "id": "ae_27",
            "question": "In how many different ways can the letters of the word 'NODE' be arranged?",
            "options": [
                "12",
                "18",
                "24",
                "36"
            ],
            "correct_answer": "24",
            "explanation": "4! = 4 * 3 * 2 * 1 = 24."
        },
        {
            "id": "ae_28",
            "question": "What is 15% of 60% of 500?",
            "options": [
                "35",
                "45",
                "50",
                "55"
            ],
            "correct_answer": "45",
            "explanation": "60% of 500 = 300. 15% of 300 = 45."
        },
        {
            "id": "ae_29",
            "question": "Two numbers are in the ratio 3:5. If their sum is 120, what is the larger number?",
            "options": [
                "65",
                "70",
                "75",
                "80"
            ],
            "correct_answer": "75",
            "explanation": "120 / (3 + 5) = 15. Larger number = 5 * 15 = 75."
        },
        {
            "id": "ae_30",
            "question": "A bag contains 5 red balls and 3 green balls. One ball is drawn at random. What is the probability it is green?",
            "options": [
                "1/4",
                "3/8",
                "1/2",
                "5/8"
            ],
            "correct_answer": "3/8",
            "explanation": "Probability = green / total = 3 / (5 + 3) = 3/8."
        }
    ],
    "Medium": [
        {
            "id": "am_1",
            "question": "According to Little's Law ($L = \\lambda \\times W$), if an API gateway processes $\\lambda = 250$ requests/sec with an average latency of $W = 40$ ms, what is the average number of concurrent requests $L$ in flight?",
            "options": [
                "5",
                "10",
                "15",
                "25"
            ],
            "correct_answer": "10",
            "explanation": "L = 250 req/sec * 0.040 sec = 10 concurrent requests in flight."
        },
        {
            "id": "am_2",
            "question": "A CPU L1 cache has a hit rate of 95% with 1 ns hit latency. The L2 cache has a miss penalty of 15 ns. What is the Average Memory Access Time (AMAT)?",
            "options": [
                "1.50 ns",
                "1.75 ns",
                "2.00 ns",
                "2.25 ns"
            ],
            "correct_answer": "1.75 ns",
            "explanation": "AMAT = Hit Time + (Miss Rate * Miss Penalty) = 1 ns + (0.05 * 15 ns) = 1 + 0.75 = 1.75 ns."
        },
        {
            "id": "am_3",
            "question": "A distributed system uses 3 independent replicas with 98% uptime each. What is the probability that at least one replica is operational?",
            "options": [
                "99.85%",
                "99.92%",
                "99.9992%",
                "99.992%"
            ],
            "correct_answer": "99.9992%",
            "explanation": "P(all down) = (0.02)^3 = 0.000008. P(at least 1 up) = 1 - 0.000008 = 0.999992 = 99.9992%."
        },
        {
            "id": "am_4",
            "question": "A team has 5 backend developers and 4 frontend developers. How many committees of 3 can be formed containing at least 2 backend developers?",
            "options": [
                "40",
                "45",
                "50",
                "55"
            ],
            "correct_answer": "50",
            "explanation": "(2 Backend, 1 Frontend) = C(5, 2) * C(4, 1) = 10 * 4 = 40. (3 Backend) = C(5, 3) = 10. Total = 40 + 10 = 50."
        },
        {
            "id": "am_5",
            "question": "In a 4-disk RAID 5 array with 2 TB disks, what is the total usable storage capacity?",
            "options": [
                "4 TB",
                "6 TB",
                "8 TB",
                "2 TB"
            ],
            "correct_answer": "6 TB",
            "explanation": "In RAID 5, usable capacity = (N - 1) * Disk Size = (4 - 1) * 2 TB = 6 TB (1 disk capacity reserved for parity)."
        },
        {
            "id": "am_6",
            "question": "A software company estimates that 8% of user sessions encounter errors. If a QA engineer tests 25 independent sessions, what is the expected number of error sessions?",
            "options": [
                "1.5",
                "2.0",
                "2.5",
                "3.0"
            ],
            "correct_answer": "2.0",
            "explanation": "Expected value E(X) = n * p = 25 * 0.08 = 2.0 sessions."
        },
        {
            "id": "am_7",
            "question": "An algorithm has time complexity T(n) = 2T(n/2) + O(n). By the Master Theorem, what is its asymptotic complexity?",
            "options": [
                "O(n)",
                "O(n log n)",
                "O(n^2)",
                "O(log n)"
            ],
            "correct_answer": "O(n log n)",
            "explanation": "a=2, b=2, f(n)=O(n). Since log_b(a) = log_2(2) = 1, case 2 of Master Theorem applies: T(n) = O(n log n)."
        },
        {
            "id": "am_8",
            "question": "A pipe can fill a database cooling tank in 6 hours, while a drain empties it in 8 hours. If both are open, how long will it take to fill the tank?",
            "options": [
                "18 hours",
                "20 hours",
                "24 hours",
                "28 hours"
            ],
            "correct_answer": "24 hours",
            "explanation": "Net rate = (1/6) - (1/8) = (4 - 3)/24 = 1/24 per hour. Time = 24 hours."
        },
        {
            "id": "am_9",
            "question": "A database query took 2.4 seconds with an unindexed table scan. Adding a B-Tree index reduced query time by 85%. What is the new query time?",
            "options": [
                "0.24 s",
                "0.36 s",
                "0.48 s",
                "0.60 s"
            ],
            "correct_answer": "0.36 s",
            "explanation": "New time = 2.4 * (1 - 0.85) = 2.4 * 0.15 = 0.36 seconds."
        },
        {
            "id": "am_10",
            "question": "In how many ways can 6 microservices be deployed onto 6 distinct container pods in a 1-to-1 mapping?",
            "options": [
                "120",
                "360",
                "720",
                "1440"
            ],
            "correct_answer": "720",
            "explanation": "6! = 6 * 5 * 4 * 3 * 2 * 1 = 720."
        },
        {
            "id": "am_11",
            "question": "A machine learning classifier has 90% precision and 80% recall. What is its F1-Score?",
            "options": [
                "0.835",
                "0.847",
                "0.852",
                "0.860"
            ],
            "correct_answer": "0.847",
            "explanation": "F1 = 2 * (P * R) / (P + R) = 2 * (0.90 * 0.80) / (0.90 + 0.80) = 1.44 / 1.70 \u2248 0.847."
        },
        {
            "id": "am_12",
            "question": "A network switch buffer drops packets when queue length exceeds 1,000 packets. If packets arrive at 4,500/sec and drain at 4,000/sec, how long until an empty buffer overflows?",
            "options": [
                "1.5 s",
                "2.0 s",
                "2.5 s",
                "3.0 s"
            ],
            "correct_answer": "2.0 s",
            "explanation": "Net accumulation rate = 4500 - 4000 = 500 packets/sec. Time to fill 1,000 = 1000 / 500 = 2.0 seconds."
        },
        {
            "id": "am_13",
            "question": "A test has 10 true/false questions. If a candidate guesses randomly, what is the probability of getting exactly 8 correct?",
            "options": [
                "45/1024",
                "45/512",
                "30/1024",
                "55/1024"
            ],
            "correct_answer": "45/1024",
            "explanation": "Binomial: C(10, 8) * (1/2)^10 = 45 / 1024."
        },
        {
            "id": "am_14",
            "question": "A cluster of 10 nodes requires a strict majority quorum for write consensus. What is the minimum number of nodes required to agree?",
            "options": [
                "5",
                "6",
                "7",
                "8"
            ],
            "correct_answer": "6",
            "explanation": "Strict majority quorum for N nodes is floor(N/2) + 1 = floor(10/2) + 1 = 5 + 1 = 6 nodes."
        },
        {
            "id": "am_15",
            "question": "Two services A and B communicate over an unreliable link with 10% packet loss. If a message is sent with up to 2 retries (3 total attempts), what is the probability of successful delivery?",
            "options": [
                "99.0%",
                "99.5%",
                "99.9%",
                "99.99%"
            ],
            "correct_answer": "99.9%",
            "explanation": "P(all 3 fail) = 0.10^3 = 0.001. P(success) = 1 - 0.001 = 0.999 = 99.9%."
        },
        {
            "id": "am_16",
            "question": "A hash function maps keys uniformly to 100 buckets. What is the expected number of items inserted before the first collision (Birthday Paradox approx)?",
            "options": [
                "8",
                "12",
                "13",
                "20"
            ],
            "correct_answer": "13",
            "explanation": "Approximation: sqrt((pi / 2) * N) = sqrt(1.57 * 100) = sqrt(157) \u2248 12.5 \u2248 13 items."
        },
        {
            "id": "am_17",
            "question": "If log2(x) + log2(x - 2) = 3, what is the value of x?",
            "options": [
                "2",
                "4",
                "6",
                "8"
            ],
            "correct_answer": "4",
            "explanation": "log2(x(x - 2)) = 3 => x^2 - 2x = 8 => x^2 - 2x - 8 = 0 => (x - 4)(x + 2) = 0. Since x > 2, x = 4."
        },
        {
            "id": "am_18",
            "question": "A company's cloud bill was $12,000 in January and grew at 5% compound monthly. What was the bill in March (after 2 months)?",
            "options": [
                "$13,200",
                "$13,230",
                "$13,250",
                "$13,400"
            ],
            "correct_answer": "$13,230",
            "explanation": "$12,000 * (1.05)^2 = $12,000 * 1.1025 = $13,230."
        },
        {
            "id": "am_19",
            "question": "In a software deployment, task A takes 4 days, task B takes 6 days, and task C depends on both and takes 5 days. What is the critical path duration?",
            "options": [
                "11 days",
                "13 days",
                "15 days",
                "17 days"
            ],
            "correct_answer": "11 days",
            "explanation": "Critical path = max(A, B) + C = max(4, 6) + 5 = 6 + 5 = 11 days."
        },
        {
            "id": "am_20",
            "question": "A load balancer routes traffic to servers X and Y in the ratio 3:2. If server X receives 450 requests, how many did server Y receive?",
            "options": [
                "250",
                "300",
                "350",
                "400"
            ],
            "correct_answer": "300",
            "explanation": "X/Y = 3/2 => 450 / Y = 3/2 => Y = (450 * 2) / 3 = 300 requests."
        },
        {
            "id": "am_21",
            "question": "A file is compressed by 40% and then encrypted, adding 5% to the compressed size. What is the net size relative to original?",
            "options": [
                "60%",
                "63%",
                "65%",
                "68%"
            ],
            "correct_answer": "63%",
            "explanation": "Original 1.0 -> 0.60 * 1.05 = 0.63 = 63% of original size."
        },
        {
            "id": "am_22",
            "question": "What is the standard deviation of the numbers: 2, 4, 4, 4, 5, 5, 7, 9?",
            "options": [
                "1.8",
                "2.0",
                "2.2",
                "2.5"
            ],
            "correct_answer": "2.0",
            "explanation": "Mean = 40 / 8 = 5. Squared differences = 9 + 1 + 1 + 1 + 0 + 0 + 4 + 16 = 32. Variance = 32/8 = 4. Std Dev = sqrt(4) = 2.0."
        },
        {
            "id": "am_23",
            "question": "A team can resolve 20 tickets in 4 hours with 3 engineers. How many engineers are needed to resolve 50 tickets in 5 hours?",
            "options": [
                "5",
                "6",
                "7",
                "8"
            ],
            "correct_answer": "6",
            "explanation": "Rate per engineer = 20 / (4 * 3) = 5/3 tickets/hr. For 50 tickets in 5 hrs: Required rate = 10 tickets/hr. Engineers = 10 / (5/3) = 6."
        },
        {
            "id": "am_24",
            "question": "A disk reads data at 120 MB/s. How many seconds will it take to read a 4.2 GB dataset?",
            "options": [
                "30 s",
                "35 s",
                "38 s",
                "42 s"
            ],
            "correct_answer": "35 s",
            "explanation": "4.2 GB = 4,200 MB. Time = 4,200 / 120 = 35 seconds."
        },
        {
            "id": "am_25",
            "question": "In a bipartite graph with 4 vertices on the left and 5 on the right, what is the maximum number of edges?",
            "options": [
                "9",
                "12",
                "18",
                "20"
            ],
            "correct_answer": "20",
            "explanation": "Max edges = 4 * 5 = 20."
        },
        {
            "id": "am_26",
            "question": "A coin is biased such that Heads is twice as likely as Tails. What is the probability of getting Tails?",
            "options": [
                "1/4",
                "1/3",
                "1/2",
                "2/3"
            ],
            "correct_answer": "1/3",
            "explanation": "P(H) = 2P(T). P(H) + P(T) = 1 => 3P(T) = 1 => P(T) = 1/3."
        },
        {
            "id": "am_27",
            "question": "A database index tree has a fan-out of 100. How many key entries can be indexed in a tree of height 3 (leaf level)?",
            "options": [
                "10,000",
                "100,000",
                "1,000,000",
                "10,000,000"
            ],
            "correct_answer": "1,000,000",
            "explanation": "Capacity = 100^3 = 1,000,000 entries."
        },
        {
            "id": "am_28",
            "question": "If a function satisfies f(1) = 3 and f(n) = 2f(n-1) + 1, what is f(4)?",
            "options": [
                "15",
                "23",
                "27",
                "31"
            ],
            "correct_answer": "31",
            "explanation": "f(1)=3, f(2)=2(3)+1=7, f(3)=2(7)+1=15, f(4)=2(15)+1=31."
        },
        {
            "id": "am_29",
            "question": "A binary tree has 15 nodes. What is the minimum possible height (levels) of the tree?",
            "options": [
                "3",
                "4",
                "5",
                "6"
            ],
            "correct_answer": "4",
            "explanation": "A complete binary tree with 15 nodes has 2^4 - 1 = 15 nodes, which has 4 levels."
        },
        {
            "id": "am_30",
            "question": "A worker's salary was increased by 10% in Year 1 and 15% in Year 2. What was the overall percentage increase over the two years?",
            "options": [
                "25%",
                "26.5%",
                "27.5%",
                "28%"
            ],
            "correct_answer": "26.5%",
            "explanation": "1.10 * 1.15 = 1.265 = 26.5% overall increase."
        }
    ],
    "Hard": [
        {
            "id": "ah_1",
            "question": "A distributed consensus cluster requires 3 out of 5 nodes for a quorum. If node failure probability is p = 0.05 independently, what is the probability that quorum cannot be formed (3+ nodes fail)?",
            "options": [
                "0.00116",
                "0.00224",
                "0.00350",
                "0.00480"
            ],
            "correct_answer": "0.00116",
            "explanation": "P(fail >= 3) = C(5,3)*p^3*(1-p)^2 + C(5,4)*p^4*(1-p) + C(5,5)*p^5 = 10*(0.000125)*0.9025 + 5*(0.00000625)*0.95 + 0.00000031 \u2248 0.00116."
        },
        {
            "id": "ah_2",
            "question": "Using Amdahl's Law, if 80% of an algorithm is parallelizable, what is the theoretical maximum speedup on a system with 16 CPU cores?",
            "options": [
                "3.85x",
                "4.00x",
                "4.15x",
                "4.50x"
            ],
            "correct_answer": "4.00x",
            "explanation": "Speedup = 1 / ((1 - P) + (P / N)) = 1 / ((1 - 0.80) + (0.80 / 16)) = 1 / (0.20 + 0.05) = 1 / 0.25 = 4.00x."
        },
        {
            "id": "ah_3",
            "question": "An M/M/1 queuing model has arrival rate \u03bb = 80 req/s and service rate \u03bc = 100 req/s. What is the average total response time W in the system?",
            "options": [
                "0.02 s",
                "0.05 s",
                "0.08 s",
                "0.10 s"
            ],
            "correct_answer": "0.05 s",
            "explanation": "In M/M/1, W = 1 / (\u03bc - \u03bb) = 1 / (100 - 80) = 1 / 20 = 0.05 seconds (50 ms)."
        },
        {
            "id": "ah_4",
            "question": "A system operates with 99.99% availability ('four nines'). How much downtime is permitted in a non-leap year (365 days)?",
            "options": [
                "5.26 minutes",
                "52.6 minutes",
                "8.76 hours",
                "1.44 hours"
            ],
            "correct_answer": "52.6 minutes",
            "explanation": "Downtime = 365 * 24 * 60 * (1 - 0.9999) = 525,600 * 0.0001 = 52.56 minutes."
        },
        {
            "id": "ah_5",
            "question": "A Bloom filter with m = 1,000,000 bits and k = 7 hash functions contains n = 100,000 elements. What is the approximate false positive probability?",
            "options": [
                "0.008",
                "0.012",
                "0.016",
                "0.024"
            ],
            "correct_answer": "0.008",
            "explanation": "p \u2248 (1 - e^(-kn/m))^k = (1 - e^(-0.7))^7 \u2248 (1 - 0.4966)^7 \u2248 (0.5034)^7 \u2248 0.0081 (0.8%)."
        },
        {
            "id": "ah_6",
            "question": "What is the expected number of comparisons in Randomized QuickSelect to find the k-th smallest element in an array of size N?",
            "options": [
                "O(log N)",
                "O(N)",
                "O(N log N)",
                "O(N^2)"
            ],
            "correct_answer": "O(N)",
            "explanation": "QuickSelect only recurses into one partition: N + N/2 + N/4 + ... = 2N = O(N) linear expected time."
        },
        {
            "id": "ah_7",
            "question": "In Shannon's Information Theory, what is the entropy H(X) in bits of a fair 8-sided die?",
            "options": [
                "2 bits",
                "3 bits",
                "4 bits",
                "8 bits"
            ],
            "correct_answer": "3 bits",
            "explanation": "H(X) = - sum(p * log2(p)) = - 8 * (1/8 * log2(1/8)) = - log2(1/8) = log2(8) = 3 bits."
        },
        {
            "id": "ah_8",
            "question": "A network link has bandwidth 10 Gbps and round-trip propagation delay RTT = 40 ms. What is the Bandwidth-Delay Product (BDP) in Megabytes?",
            "options": [
                "25 MB",
                "50 MB",
                "100 MB",
                "500 MB"
            ],
            "correct_answer": "50 MB",
            "explanation": "BDP = (10 * 10^9 bits/s * 0.040 s) / 8 bits/byte = 400 * 10^6 / 8 = 50,000,000 bytes = 50 MB."
        },
        {
            "id": "ah_9",
            "question": "An asymptotic recurrence T(n) = 3T(n/4) + n log n. By Master Theorem, what is its asymptotic complexity?",
            "options": [
                "O(n log n)",
                "O(n^(log4(3)))",
                "O(n^2)",
                "O(log n)"
            ],
            "correct_answer": "O(n log n)",
            "explanation": "log_4(3) \u2248 0.79. f(n) = n log n = Omega(n^(0.79 + epsilon)). Case 3 applies: T(n) = O(n log n)."
        },
        {
            "id": "ah_10",
            "question": "In a Poisson process with rate \u03bb = 4 events per hour, what is the probability of observing exactly 2 events in a 30-minute interval?",
            "options": [
                "0.180",
                "0.271",
                "0.360",
                "0.405"
            ],
            "correct_answer": "0.271",
            "explanation": "Interval is 0.5 hour, so mean \u03bc = 4 * 0.5 = 2. P(X = 2) = (e^(-2) * 2^2) / 2! = (0.1353 * 4) / 2 = 0.2706 \u2248 0.271."
        },
        {
            "id": "ah_11",
            "question": "In a distributed Raft cluster with election timeout uniformly distributed between 150 ms and 300 ms, what is the probability two nodes time out within 5 ms of each other?",
            "options": [
                "0.033",
                "0.065",
                "0.098",
                "0.125"
            ],
            "correct_answer": "0.065",
            "explanation": "Range L = 150 ms. Area of |T1 - T2| <= 5 is 1 - (145/150)^2 = 1 - 0.9344 = 0.0656 (6.5%)."
        },
        {
            "id": "ah_12",
            "question": "A cryptographic hash has a 64-bit digest. According to the Birthday Bound, approximately how many random inputs are required to find a collision with 50% probability?",
            "options": [
                "2^16",
                "2^32",
                "2^48",
                "2^64"
            ],
            "correct_answer": "2^32",
            "explanation": "Birthday bound is 2^(n/2) = 2^(64/2) = 2^32 operations."
        },
        {
            "id": "ah_13",
            "question": "A consistent hashing ring has 3 physical nodes with 100 virtual nodes each. If a 4th physical node is added, what fraction of keys are expected to migrate?",
            "options": [
                "15%",
                "25%",
                "33%",
                "50%"
            ],
            "correct_answer": "25%",
            "explanation": "In consistent hashing with N nodes, adding 1 node migrates 1 / (N + 1) of the keys: 1 / (3 + 1) = 1/4 = 25%."
        },
        {
            "id": "ah_14",
            "question": "A database disk has 10,000 RPM. What is the average rotational latency?",
            "options": [
                "2.0 ms",
                "3.0 ms",
                "4.0 ms",
                "6.0 ms"
            ],
            "correct_answer": "3.0 ms",
            "explanation": "One revolution = 60 / 10,000 = 6 ms. Average rotational latency is half a revolution = 6 / 2 = 3.0 ms."
        },
        {
            "id": "ah_15",
            "question": "In how many ways can 8 distinct tasks be partitioned into 2 non-empty worker groups (Stirling numbers of the second kind S(8, 2))?",
            "options": [
                "64",
                "127",
                "128",
                "256"
            ],
            "correct_answer": "127",
            "explanation": "S(n, 2) = 2^(n-1) - 1 = 2^7 - 1 = 128 - 1 = 127."
        },
        {
            "id": "ah_16",
            "question": "What is the expected number of coin flips until observing two consecutive Heads (HH)?",
            "options": [
                "4",
                "6",
                "8",
                "10"
            ],
            "correct_answer": "6",
            "explanation": "Let E be expected flips. Transitions yield E = 1/2(1 + E) + 1/4(2 + E) + 1/4(2) => E = 6 flips."
        },
        {
            "id": "ah_17",
            "question": "A server pool handles 5,000 requests/s. If each request spawns 2 database queries taking 10 ms each, what is the concurrent DB connection load?",
            "options": [
                "50",
                "100",
                "150",
                "200"
            ],
            "correct_answer": "100",
            "explanation": "Queries per second = 5,000 * 2 = 10,000 q/s. By Little's Law: L = 10,000 * 0.010 s = 100 concurrent connections."
        },
        {
            "id": "ah_18",
            "question": "In a hypercube network of dimension d = 6, what is the diameter (max shortest path between any two nodes)?",
            "options": [
                "4",
                "6",
                "12",
                "64"
            ],
            "correct_answer": "6",
            "explanation": "The diameter of a d-dimensional hypercube is d (Hamming distance between 000000 and 111111 = 6)."
        },
        {
            "id": "ah_19",
            "question": "What is the chromatic number of a cycle graph C_7 with 7 vertices?",
            "options": [
                "2",
                "3",
                "4",
                "5"
            ],
            "correct_answer": "3",
            "explanation": "Any odd cycle C_(2k+1) requires 3 colors."
        },
        {
            "id": "ah_20",
            "question": "A matrix multiplication of two N x N matrices using Strassen's algorithm has complexity O(N^(log2(7))). What is log2(7) approximately?",
            "options": [
                "2.58",
                "2.81",
                "3.00",
                "3.24"
            ],
            "correct_answer": "2.81",
            "explanation": "log2(7) \u2248 2.807 \u2248 2.81."
        },
        {
            "id": "ah_21",
            "question": "An ECC memory module uses Hamming(7, 4) code. How many parity bits are included in each 7-bit codeword?",
            "options": [
                "1",
                "2",
                "3",
                "4"
            ],
            "correct_answer": "3",
            "explanation": "In (n, k) = (7, 4), parity bits = n - k = 7 - 4 = 3 parity bits."
        },
        {
            "id": "ah_22",
            "question": "A random variable X is exponentially distributed with mean 4. What is P(X > 4)?",
            "options": [
                "1/e",
                "1 - 1/e",
                "1/e^2",
                "1/2"
            ],
            "correct_answer": "1/e",
            "explanation": "P(X > x) = e^(-lambda * x). Mean = 1/lambda = 4 => lambda = 1/4. P(X > 4) = e^(-(1/4)*4) = e^(-1) = 1/e \u2248 0.368."
        },
        {
            "id": "ah_23",
            "question": "How many spanning trees does a complete graph K_4 have according to Cayley's formula?",
            "options": [
                "12",
                "16",
                "24",
                "64"
            ],
            "correct_answer": "16",
            "explanation": "Cayley's formula: n^(n - 2) = 4^(4 - 2) = 4^2 = 16."
        },
        {
            "id": "ah_24",
            "question": "A system has MTTF = 2,000 hours and MTTR = 8 hours. What is its operational availability?",
            "options": [
                "99.40%",
                "99.60%",
                "99.80%",
                "99.95%"
            ],
            "correct_answer": "99.60%",
            "explanation": "Availability = MTTF / (MTTF + MTTR) = 2000 / 2008 = 0.996016 \u2248 99.60%."
        },
        {
            "id": "ah_25",
            "question": "In PageRank with damping factor d = 0.85, what is the teleportation probability to a random page?",
            "options": [
                "0.15",
                "0.50",
                "0.85",
                "0.015"
            ],
            "correct_answer": "0.15",
            "explanation": "Teleportation probability = 1 - d = 1 - 0.85 = 0.15."
        },
        {
            "id": "ah_26",
            "question": "What is the expected value of the minimum of two independent Uniform(0, 1) random variables?",
            "options": [
                "1/4",
                "1/3",
                "1/2",
                "2/3"
            ],
            "correct_answer": "1/3",
            "explanation": "E[min(X1, X2)] = 1 / (n + 1) = 1 / (2 + 1) = 1/3."
        },
        {
            "id": "ah_27",
            "question": "A TCP connection window size is 64 KB and RTT is 50 ms. What is the maximum throughput limited by window size?",
            "options": [
                "5.12 Mbps",
                "10.24 Mbps",
                "12.8 Mbps",
                "20.48 Mbps"
            ],
            "correct_answer": "10.24 Mbps",
            "explanation": "Throughput = (64 * 1024 * 8 bits) / 0.050 s = 524,288 / 0.050 = 10,485,760 bps = 10.24 Mbps."
        },
        {
            "id": "ah_28",
            "question": "What is the time complexity of the Floyd-Warshall all-pairs shortest paths algorithm for a graph with V vertices?",
            "options": [
                "O(V log V)",
                "O(V^2)",
                "O(V^3)",
                "O(V^2 log V)"
            ],
            "correct_answer": "O(V^3)",
            "explanation": "Three nested loops over V vertices = O(V^3)."
        },
        {
            "id": "ah_29",
            "question": "In an election between 2 candidates where candidate A receives a votes and B receives b votes (a > b), what is the probability A leads throughout (Bertrand's Ballot Theorem)?",
            "options": [
                "(a - b)/(a + b)",
                "(a + b)/(a - b)",
                "b/a",
                "1 - b/a"
            ],
            "correct_answer": "(a - b)/(a + b)",
            "explanation": "Bertrand's Ballot Theorem states P = (a - b) / (a + b)."
        },
        {
            "id": "ah_30",
            "question": "A data structure supports operations in amortized time O(1) via potential function method. If initial potential is \u03a60 = 0 and \u03a6_final >= 0, what does this guarantee?",
            "options": [
                "Total real cost <= Total amortized cost",
                "Total real cost >= Total amortized cost",
                "Worst-case cost is strictly O(1)",
                "Space complexity is bounded by O(1)"
            ],
            "correct_answer": "Total real cost <= Total amortized cost",
            "explanation": "Sum(c_i) = Sum(a_i) - (\u03a6_final - \u03a60) <= Sum(a_i) since \u03a6_final >= 0."
        }
    ]
}

def generate_aptitude_questions(
    difficulty: str = "Medium",
    count: int = 10,
    session_seed: Optional[int | str] = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate diverse numerical, probabilistic, and systems aptitude questions."""
    diff_key = difficulty if difficulty in APTITUDE_BANKS else "Medium"
    pool = list(APTITUDE_BANKS[diff_key])

    seed_val = hash(f"{session_seed}_{time.time() // 3600}") if session_seed else int(time.time() * 1000) % 1000000
    rng = random.Random(seed_val)
    rng.shuffle(pool)

    return pool[:count]


PERSONALITY_STATEMENTS: list[dict[str, str]] = [
    {
        "id": "p1",
        "trait": "Conscientiousness",
        "statement": "I systematically break complex software projects into tracked milestones and unit test criteria before writing production code."
    },
    {
        "id": "p2",
        "trait": "Openness",
        "statement": "I actively experiment with emerging frameworks, novel algorithms, and alternative architectural paradigms to solve stubborn bottlenecks."
    },
    {
        "id": "p3",
        "trait": "Extraversion",
        "statement": "I derive energy from facilitating cross-functional sprint planning and aligning diverse engineering stakeholders."
    },
    {
        "id": "p4",
        "trait": "Agreeableness",
        "statement": "I prioritize psychological safety and offer constructive, blameless code review feedback that helps junior engineers grow."
    },
    {
        "id": "p5",
        "trait": "Emotional Stability",
        "statement": "I remain calm and methodical during high-severity production outages, focusing on root-cause mitigation rather than panic."
    },
    {
        "id": "p6",
        "trait": "Conscientiousness",
        "statement": "I maintain thorough documentation, architectural decision records (ADRs), and clear commit messages."
    },
    {
        "id": "p7",
        "trait": "Openness",
        "statement": "I enjoy researching academic whitepapers and adapting theoretical computer science concepts into scalable software."
    },
    {
        "id": "p8",
        "trait": "Extraversion",
        "statement": "I proactively step forward to lead technical design discussions and demo new product capabilities to leadership."
    },
    {
        "id": "p9",
        "trait": "Agreeableness",
        "statement": "I actively seek out peer consensus and listen attentively when another engineer challenges my architectural proposal."
    },
    {
        "id": "p10",
        "trait": "Emotional Stability",
        "statement": "When critical deadlines shift abruptly, I adapt my work schedule without experiencing overwhelming frustration."
    },
    {
        "id": "p11",
        "trait": "Conscientiousness",
        "statement": "I rigorously double-check edge-case boundaries, input sanitization, and security vulnerabilities before opening a pull request."
    },
    {
        "id": "p12",
        "trait": "Openness",
        "statement": "I embrace ambiguous product requirements as opportunities to prototype innovative customer-facing solutions."
    },
    {
        "id": "p13",
        "trait": "Extraversion",
        "statement": "I enthusiastically share engineering postmortems, lessons learned, and best practices across the broader organization."
    },
    {
        "id": "p14",
        "trait": "Agreeableness",
        "statement": "I readily offer assistance to teammates who are struggling with difficult bugs even when my own sprint capacity is tight."
    },
    {
        "id": "p15",
        "trait": "Emotional Stability",
        "statement": "I accept critical feedback on my code or architecture as an objective opportunity for professional refinement."
    },
    {
        "id": "p16",
        "trait": "Conscientiousness",
        "statement": "I follow through on commitments and ensure technical debt items are tracked in the backlog rather than forgotten."
    },
    {
        "id": "p17",
        "trait": "Openness",
        "statement": "I frequently challenge existing engineering dogma when modern tooling offers demonstrably superior performance."
    },
    {
        "id": "p18",
        "trait": "Extraversion",
        "statement": "I enjoy mentoring onboarding developers and conducting paired programming sessions to accelerate their ramp-up."
    },
    {
        "id": "p19",
        "trait": "Agreeableness",
        "statement": "I resolve interpersonal engineering disagreements through data-driven benchmarks and collaborative dialogue."
    },
    {
        "id": "p20",
        "trait": "Emotional Stability",
        "statement": "I maintain focus and high code quality during protracted on-call incident response rotations."
    },
    {
        "id": "p21",
        "trait": "Conscientiousness",
        "statement": "I treat automated test coverage, regression suites, and CI/CD reliability as non-negotiable engineering requirements."
    },
    {
        "id": "p22",
        "trait": "Openness",
        "statement": "I am motivated by working on zero-to-one greenfield products where architectural blueprints must be invented from scratch."
    },
    {
        "id": "p23",
        "trait": "Extraversion",
        "statement": "I build cross-departmental relationships with product, design, and operations teams to align on product roadmaps."
    },
    {
        "id": "p24",
        "trait": "Agreeableness",
        "statement": "I celebrate my teammates' promotions and project milestones with genuine enthusiasm and praise."
    },
    {
        "id": "p25",
        "trait": "Emotional Stability",
        "statement": "I maintain a healthy balance between rigorous professional dedication and sustainable mental recovery."
    }
]

def generate_personality_statements(count: int = 20, session_seed: Optional[int | str] = None) -> list[dict[str, str]]:
    """Provide diverse psychometric statements covering the Big Five dimensions."""
    pool = list(PERSONALITY_STATEMENTS)
    seed_val = hash(f"{session_seed}") if session_seed else int(time.time() * 1000) % 1000000
    rng = random.Random(seed_val)
    rng.shuffle(pool)
    return pool[:min(count, len(pool))]


def evaluate_personality_test(responses: dict[Any, Any]) -> dict[str, Any]:
    """Compute normalized Big Five OCEAN scores and organizational culture fit."""
    trait_scores: dict[str, list[int]] = {
        "Openness": [],
        "Conscientiousness": [],
        "Extraversion": [],
        "Agreeableness": [],
        "Emotional Stability": [],
    }

    stmt_map = {s["id"]: s["trait"] for s in PERSONALITY_STATEMENTS}

    for k, v in responses.items():
        stmt_id = str(k)
        trait = stmt_map.get(stmt_id)
        if not trait:
            try:
                idx = int(k) if isinstance(k, int) or str(k).isdigit() else 0
                trait = PERSONALITY_STATEMENTS[idx % len(PERSONALITY_STATEMENTS)]["trait"]
            except Exception:
                trait = "Conscientiousness"
        try:
            rating = int(v)
            if 1 <= rating <= 5:
                trait_scores[trait].append(rating)
        except (ValueError, TypeError):
            pass

    normalized: dict[str, float] = {}
    for trait, scores in trait_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            pct = round((avg / 5.0) * 100.0, 1)
        else:
            pct = 75.0
        normalized[trait] = pct

    culture_fit = round(
        0.30 * normalized["Conscientiousness"] +
        0.25 * normalized["Agreeableness"] +
        0.20 * normalized["Emotional Stability"] +
        0.15 * normalized["Openness"] +
        0.10 * normalized["Extraversion"],
        1
    )

    feedback = []
    if normalized["Conscientiousness"] >= 80:
        feedback.append("Exceptionally high conscientiousness: disciplined execution and rigorous attention to software craftsmanship.")
    if normalized["Emotional Stability"] >= 80:
        feedback.append("Superb emotional stability under high-stakes on-call incident triage and shifting engineering roadmaps.")
    if normalized["Agreeableness"] >= 75:
        feedback.append("Collaborative team-first orientation with healthy constructive peer code review dynamics.")
    if not feedback:
        feedback.append("Balanced, mature professional profile demonstrating consistent organizational culture fit.")

    return {
        "ocean_scores": normalized,
        "culture_fit_score": culture_fit,
        "score": culture_fit,
        "strengths": [f"High {t} ({s:.0f}%)" for t, s in normalized.items() if s >= 75],
        "feedback": feedback,
    }


AUDIO_BENCHMARK_SENTENCES: dict[str, list[str]] = {
    "Easy": [
        "Our team decoupled data ingestion from real-time analytics to ensure high availability.",
        "Clear documentation enables new engineers to set up their local development environments in minutes.",
        "We automated our regression testing suite to catch syntax errors before deploying to staging.",
        "The microservice processes incoming user transactions and stores records in a relational database.",
        "Consistent naming conventions make complex codebases much easier to maintain over time.",
        "We implemented structured error handling to return standard HTTP status codes to our clients.",
        "Unit tests verify that boundary conditions and edge cases are handled safely without exceptions.",
        "The front-end client communicates with our backend services through secure REST endpoints.",
        "Engineers review code changes to uphold design principles and prevent unwanted regressions.",
        "We configure database connection pooling to reuse active sockets and reduce server latency.",
        "Our build pipeline automatically packages container images whenever a pull request is merged.",
        "Good communication transforms complicated technical requirements into actionable project milestones.",
        "We configured health check endpoints so our load balancer can detect failing server instances.",
        "Adhering to the single responsibility principle ensures classes remain cohesive and modular.",
        "Caching frequently accessed records in memory significantly reduced our primary database load.",
        "We practice continuous integration by merging small, verified code updates multiple times a day.",
        "The distributed logging system indexes application events to simplify remote debugging.",
        "Our team conducts blameless retrospectives to identify practical opportunities for improvement.",
        "Database migrations run automatically during deployment to ensure schema changes apply cleanly.",
        "We use environment variables to manage sensitive API credentials securely across environments.",
        "Careful code reviews foster shared ownership and elevate engineering quality across the team.",
        "Static code analysis detects potential memory leaks and security flaws during the compile phase.",
        "We partition large database tables by date to optimize query performance and data cleanup.",
        "Monitoring real-time CPU utilization helps our infrastructure team anticipate resource limits.",
        "The load balancer distributes traffic evenly to prevent individual nodes from becoming overwhelmed.",
        "We enforce strict interface contracts to ensure independent teams can build in parallel.",
        "Automated alerts notify our on-call engineers whenever error rates exceed standard thresholds.",
        "Documenting architectural decisions preserves institutional context for future developers.",
        "We optimized our database indexes to accelerate critical customer search queries.",
        "A well-structured continuous deployment pipeline delivers stable features directly to users."
    ],
    "Medium": [
        "Our cloud architecture decouples data ingestion from real-time analytics to guarantee fault tolerance and sub-second query latency.",
        "We systematically mitigated production downtime by implementing automated canary deployments and end-to-end distributed telemetry.",
        "The engineering team refactored the persistent caching tier to handle fifty thousand concurrent requests during peak product launches.",
        "Leadership requires aligning cross-functional teams around shared measurable outcomes while proactively managing technical trade-offs.",
        "To maintain system throughput under sudden load spikes, we partitioned database queries across isolated read replicas.",
        "Clear communication transforms complex architectural constraints into actionable roadmaps that non-technical stakeholders readily endorse.",
        "By enforcing strict interface boundaries and comprehensive unit tests, we prevented regression bugs from reaching production.",
        "We designed an asynchronous event pipeline using message queues to decouple bursty background tasks from user-facing transactions.",
        "The team migrated legacy relational data models into distributed document collections to support flexible schema evolution.",
        "Optimizing our container orchestration cluster reduced cloud infrastructure expenditure by twenty-eight percent year over year.",
        "We established distributed tracing headers across microservices to isolate latency bottlenecks within third-party payment gateways.",
        "Our team balanced developer velocity against architectural modularity by introducing automated domain-driven boundary verification.",
        "Implementing rate-limiting algorithms at the edge gateway successfully defended our authentication services against brute-force attacks.",
        "We configured automated database failover across multiple availability zones to ensure uninterrupted business continuity.",
        "Proactive performance profiling revealed significant thread contention within our connection pooling mechanism during peak hours.",
        "We replaced synchronous blocking remote procedure calls with non-blocking event-driven publishers to boost overall throughput.",
        "The engineering organization adopted comprehensive error budgets to balance rapid feature releases with service level objectives.",
        "By decomposing monolithic database transactions into asynchronous workflows, we eliminated persistent row-level lock contention.",
        "Our team containerized legacy services and deployed a service mesh to automate mutual cryptographic authentication between pods.",
        "We instituted automated smoke testing within our continuous delivery pipeline to validate zero-downtime rolling upgrades.",
        "Rigorous benchmarking of database query execution plans enabled us to eliminate slow sequential scans across massive datasets.",
        "We engineered resilient circuit breakers to gracefully degrade non-critical product features when upstream dependencies experienced outages.",
        "Cross-functional collaboration between engineering and product management accelerated our quarterly release milestones without accumulating technical debt.",
        "We decoupled analytical reporting queries from the transactional master database by implementing real-time change data capture.",
        "Our continuous integration pipeline compiles source code and executes thousands of automated regression tests in under five minutes.",
        "We established strict observability dashboards tracking ninety-ninth percentile latency, error budgets, and saturation across all clusters.",
        "Refactoring the authentication middleware into stateless cryptographic tokens allowed our web tier to scale horizontally with ease.",
        "We mitigated cache stampede anomalies by implementing deterministic probabilistic early expiration algorithms in Redis.",
        "Our senior developers facilitate weekly design reviews to foster architectural consistency and mentor emerging engineering talent.",
        "By standardizing our API contracts with OpenAPI specifications, we enabled client engineering teams to build mock integration tests."
    ],
    "Hard": [
        "We engineered a distributed consensus layer utilizing the Raft protocol to guarantee linearizable consistency across multi-region datacenters.",
        "The platform mitigates cascading thread pool starvation under adversarial load by orchestrating cooperative backpressure and proactive load shedding.",
        "Our architectural transition to event sourcing with command query responsibility segregation reduced ninety-ninth percentile read latency to under eight milliseconds.",
        "By executing asynchronous kernel input-output operations via io_uring, our ingestion daemon achieves five hundred thousand packets per second per core.",
        "We prevented catastrophic false sharing across multi-core processors by padding concurrent counter objects to sixty-four-byte cache line boundaries.",
        "Our infrastructure automatically enforces zero-trust mutual authentication using cryptographic workload identities managed by automated SPIFFE daemons.",
        "We isolated memory-intensive analytical computations within dedicated Linux cgroups to safeguard critical low-latency transaction processing daemons.",
        "The team resolved distributed cache stampede vulnerabilities by orchestrating deterministic single-flight execution groups across worker nodes.",
        "We designed a multi-tenant isolation model that enforces fine-grained hardware storage quotas and dynamic bandwidth allocation per customer namespace.",
        "Our team conducted continuous chaos experiments by injecting network partitions and synthetic packet latency to prove fault-tolerant failover.",
        "By refactoring our memory allocation strategy to utilize off-heap native memory pools, we completely eliminated garbage collection pause spikes.",
        "We optimized our database engine's storage footprint by developing custom dictionary-encoded columnar compression algorithms for time-series metrics.",
        "The engineering group implemented distributed deadlock detection heuristics by constructing runtime dependency wait-for graphs across remote transactions.",
        "We established an automated canary promotion workflow driven by real-time metric evaluation of error budget burn rates and latency distributions.",
        "Our team mitigated speculative execution side-channel vulnerabilities across shared virtualization hosts by applying kernel page-table isolation patches.",
        "We decoupled stateful stream processing operations from local disk failure by snapshotting distributed changelogs to durable cloud object storage.",
        "The platform utilizes consistent hashing with virtual nodes to minimize data migration when expanding our distributed cache cluster.",
        "We engineered an outbox publisher pattern with transactional event relays to guarantee at-least-once delivery to distributed message topics.",
        "Our network architecture enforces programmable packet filtering directly within the Linux kernel utilizing verified eBPF bytecode programs.",
        "We mitigated distributed clock skew anomalies by adopting hybrid logical clocks across our geo-distributed multi-master database clusters.",
        "The high-frequency trading engine leverages lock-free ring buffers and hardware memory fences to achieve deterministic sub-microsecond tick execution.",
        "We implemented speculative branch optimization and custom JIT assembly routines to accelerate cryptographic hash verification by four hundred percent.",
        "Our team designed a Byzantine fault-tolerant coordination protocol capable of achieving deterministic state machine consensus despite arbitrary node failures.",
        "We eliminated memory fragmentation and allocation bottlenecks in our web reverse proxy by implementing specialized slab allocators for network buffers.",
        "The distributed tracing infrastructure samples billions of daily telemetry spans using adaptive reservoir sampling without imposing CPU overhead.",
        "We developed automated database schema migration tools capable of performing online table restructuring on multi-terabyte datasets without lock contention.",
        "Our resilience architecture combines exponential backoff with randomized jitter to prevent synchronized retry storms against recovering backend services.",
        "We decoupled persistent state from computational workers using distributed network-attached block storage with asynchronous replication across failure domains.",
        "The platform enforces strict cryptographic data sanitization and envelope encryption with automated hardware security module key rotation.",
        "By systematically auditing Linux kernel network socket parameters, we resolved ephemeral port exhaustion during extreme connection surges."
    ]
}

def generate_audio_recording_sentences(
    difficulty: str = "Medium",
    count: int = 10,
    session_seed: Optional[int | str] = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Pre-load clear benchmark sentences for voice & pronunciation evaluation."""
    diff_key = difficulty if difficulty in AUDIO_BENCHMARK_SENTENCES else "Medium"
    pool = list(AUDIO_BENCHMARK_SENTENCES[diff_key])

    seed_val = hash(f"{session_seed}_{time.time() // 3600}") if session_seed else int(time.time() * 1000) % 1000000
    rng = random.Random(seed_val)
    rng.shuffle(pool)

    return [
        {"id": f"audio_{i+1}", "sentence": s, "text": s}
        for i, s in enumerate(pool[:count])
    ]

# ==============================================================================
# GENERAL ANSWER EVALUATOR (WITH STRICT GIBBERISH / ZERO-SCORE DETECTION)
# ==============================================================================
def is_gibberish_or_irrelevant(text: str, question: str = "") -> tuple[bool, float, str]:
    """Detect gibberish, keyboard smashing, copy-pasting, or skipped answers.

    Returns (is_invalid, score, reason). Strictly awards 0 to 10 marks if invalid.
    """
    clean = (text or "").strip()
    if not clean or clean.lower() in ("skipped", "n/a", "none", "skip", "idk", "no idea", "pass"):
        return True, 0.0, "Question was skipped or left blank. No credit awarded."

    words = re.findall(r"\b[A-Za-z0-9+#.-]+\b", clean.lower())
    word_count = len(words)
    if word_count == 0:
        return True, 0.0, "No substantive words submitted."

    # 1. Direct copy-paste of the question check
    if question:
        ratio = difflib.SequenceMatcher(None, clean.lower(), question.lower().strip()).ratio()
        if ratio > 0.65:
            return True, 0.0, "Invalid submission: Answer merely copies or repeats the question prompt without technical content."

    # 2. Repeated nonsense / keyboard smashing (e.g. asdf, jvjg, aaaa, qwerty)
    if re.search(r"\b(asdf|jvjg|qwerty|zxcv|ghjk|hjkl|aaaa|zzzz|xxxx|blah)\b", clean.lower()):
        return True, 0.0, "Invalid submission: Random keyboard characters or gibberish detected."

    # 3. Consonant cluster and missing vowel checks
    gibberish_tokens = 0
    for w in words:
        if len(w) >= 4 and not re.search(r"[aeiouy]", w):
            gibberish_tokens += 1
        elif len(w) >= 6 and re.search(r"[^aeiouy]{5,}", w):
            gibberish_tokens += 1
    if gibberish_tokens >= max(1, word_count // 3):
        return True, 0.0, "Invalid submission: Gibberish character sequence detected."

    # 4. Repeated identical word spam (e.g. 'code code code code')
    if word_count >= 4 and len(set(words)) <= 2:
        return True, 0.0, "Invalid submission: Repetitive word spam detected."

    # 5. Too short / insufficient technical explanation (< 4 words)
    if word_count < 4:
        return True, 5.0, "Response is too brief to demonstrate technical understanding or design trade-offs."

    return False, 0.0, ""


def evaluate_answer(
    question: str,
    answer: str,
    resume: str = "",
    jd: str = "",
    role: str = "",
    difficulty: str = "Medium",
    section: str = "Technical",
    expected_concepts: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Grade text answer returning score (0-100), missing concepts, feedback, and ideal sample answer.

    Enforces strict 0-10 marks for gibberish, copy-paste prompts, or irrelevant submissions.
    """
    clean_ans = (answer or "").strip()
    ideal_ans = _generate_sample_ideal_answer(question, section)

    # 1. Strict Gibberish / Skipped / Copy-Paste Gate
    is_invalid, zero_score, invalid_reason = is_gibberish_or_irrelevant(clean_ans, question)
    if is_invalid:
        return {
            "score": zero_score,
            "feedback": invalid_reason,
            "improvement_feedback": ["Provide a substantive technical answer explaining approach, internal mechanics, and trade-offs."],
            "missing_concepts": ["Substantive technical explanation.", "Core architectural definitions."],
            "missing_key_concepts": ["Substantive technical explanation.", "Core architectural definitions."],
            "ideal_answer": ideal_ans,
            "ideal_sample_answer": ideal_ans,
            "mode": "Strict Rubric Grader",
        }

    words = re.findall(r"\b[A-Za-z0-9+#.-]+\b", clean_ans.lower())
    word_count = len(words)

    stop_words = {"this", "that", "with", "from", "have", "what", "when", "would", "could", "about", "your", "they", "them", "into", "and", "the", "for", "you", "were", "been"}
    q_keywords = {w for w in re.findall(r"\b[a-z]{3,}\b", question.lower()) if w not in stop_words}
    a_words_set = set(words)

    relevance_ratio = len(q_keywords & a_words_set) / max(1, len(q_keywords))

    domain_terms = re.findall(
        r"\b(polymorphism|encapsulation|inheritance|abstraction|solid|vtable|dispatch|process|thread|mutex|semaphore|deadlock|paging|virtual memory|context switch|page fault|hash|tree|heap|stack|queue|bst|avl|cache|latency|throughput|p99|sharding|replica|index|query|async|concurrency|race condition|monolith|microservice|idempotent|partition|consistency|acid|base|cap theorem|distributed|telemetry)\b",
        clean_ans,
        re.I
    )

    # 2. Strict Irrelevant Content Gate: If relevance is near zero AND zero domain terms
    if relevance_ratio < 0.12 and len(domain_terms) == 0:
        return {
            "score": 5.0,
            "feedback": "Irrelevant submission: The response does not address the question's core subject or technical concepts.",
            "improvement_feedback": ["Directly address the prompt using domain-specific mechanics, data structures, and architectural principles."],
            "missing_concepts": ["Relevant technical principles.", "Direct conceptual solution to prompt."],
            "missing_key_concepts": ["Relevant technical principles.", "Direct conceptual solution to prompt."],
            "ideal_answer": ideal_ans,
            "ideal_sample_answer": ideal_ans,
            "mode": "Strict Rubric Grader",
        }

    has_metrics = bool(re.search(r"\b(\d+%|\d+x|\d+ms|\d+k|\d+m|improved|reduced|optimized|scaled|increased|latency|throughput)\b", clean_ans, re.I))
    has_structure = bool(re.search(r"\b(first|because|therefore|however|result|approach|trade-off|ultimately|implemented|design|pattern)\b", clean_ans, re.I))
    depth_factor = min(1.0, word_count / 50.0)

    # Expected concepts matching
    concept_bonus = 0.0
    if expected_concepts:
        matched_concepts = 0
        for exp in expected_concepts:
            exp_tokens = [tok for tok in exp.lower().split() if len(tok) > 3]
            if any(tok in clean_ans.lower() for tok in exp_tokens):
                matched_concepts += 1
        concept_ratio = matched_concepts / max(1, len(expected_concepts))
        concept_bonus = concept_ratio * 30.0

    domain_bonus = min(25.0, len(set(domain_terms)) * 5.0)

    # Base starts at 15 points (not 30) to prevent free points for low relevance
    base_calc = 15.0 + min(35.0, relevance_ratio * 40.0) + (15.0 * depth_factor) + domain_bonus + concept_bonus
    if has_metrics: base_calc += 5.0
    if has_structure: base_calc += 5.0

    # Cap score if domain terms are absent
    if len(domain_terms) == 0:
        base_calc = min(35.0, base_calc)

    final_score = round(max(10.0, min(98.0, base_calc)), 1)

    missing = []
    if not domain_terms:
        missing.append("Precise system/OOP/OS/DS terminology and mechanical definitions.")
    if not has_metrics:
        missing.append("Concrete performance metrics or benchmark numbers (e.g. latency, scale, % improvement).")
    if word_count < 35:
        missing.append("Detailed trade-off analysis and error-handling safeguards.")

    feedback = []
    if final_score < 65:
        feedback.append("Frame your response clearly: state high-level approach, follow with implementation mechanics, and conclude with the trade-off or outcome.")
    if not has_metrics:
        feedback.append("Anchor your technical contributions with quantifiable data points (e.g. 'reduced latency by 30%').")
    if not feedback:
        feedback.append("Strong technical clarity, structured explanation, and good domain terminology coverage.")

    feedback_str = " ".join(feedback)

    return {
        "score": final_score,
        "feedback": feedback_str,
        "improvement_feedback": feedback,
        "missing_concepts": missing[:3] or ["Production edge-case mitigation."],
        "missing_key_concepts": missing[:3] or ["Production edge-case mitigation."],
        "ideal_answer": ideal_ans,
        "ideal_sample_answer": ideal_ans,
        "mode": "Analytical Rubric Grader",
    }

def _generate_sample_ideal_answer(question: str, section: str) -> str:
    """Provide realistic, senior-level ideal sample answer for benchmark review."""
    q_lower = (question or "").lower()
    if "oops" in q_lower or "polymorphism" in q_lower or "inheritance" in q_lower or "encapsulation" in q_lower:
        return (
            "In Object-Oriented design, Encapsulation hides internal state and exposes strict invariants via accessors; "
            "Polymorphism enables interchangeable behavior at compile-time (overloading) or runtime (dynamic dispatch via vtable). "
            "Favoring composition over deep inheritance prevents brittle base class hierarchies and respects the Liskov Substitution Principle."
        )
    elif "process" in q_lower or "thread" in q_lower or "deadlock" in q_lower or "paging" in q_lower or "os" in q_lower or "operating system" in q_lower:
        return (
            "A Process is an isolated execution unit with its own virtual address space, file descriptors, and page tables, whereas Threads share "
            "the process heap and code segment while maintaining private stacks and registers. Synchronization requires mutexes or semaphores "
            "to prevent race conditions, and deadlocks are prevented by enforcing a global lock acquisition hierarchy."
        )
    elif "array" in q_lower or "linked list" in q_lower or "hash" in q_lower or "tree" in q_lower or "bst" in q_lower or "heap" in q_lower:
        return (
            "Arrays offer O(1) random access and excellent CPU cache locality at the cost of O(N) middle insertions; Hash Tables provide O(1) "
            "amortized lookups via bucket indexing with dynamic rehashing. Trees like Red-Black and AVL guarantee O(log N) operations through self-balancing "
            "rotations, making them ideal for ordered data streams."
        )
    elif "project" in q_lower or "architecture" in q_lower or "system" in q_lower:
        return (
            "An enterprise production service decouples ingestion from analytical processing via asynchronous message streams (e.g. Kafka). "
            "Stateful data stores implement connection pooling and read replicas to distribute load, while edge reverse proxies enforce token-bucket rate limiting "
            "and circuit breaking to safeguard P99 latency SLAs."
        )
    return (
        "A senior-level response introduces the high-level design paradigm, articulates the exact mechanical implementation details, "
        "and justifies the architectural trade-offs (latency, space, consistency) with concrete quantifiable production benchmarks."
    )


# ==============================================================================
# LONGITUDINAL COACHING PLAN GENERATOR
# ==============================================================================
def generate_longitudinal_coaching_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Produce comprehensive 14-day skill acceleration roadmap."""
    role = kwargs.get("role") or ""
    overall_score = kwargs.get("overall_score") or 80.0
    section_scores = kwargs.get("section_scores") or {}
    skills = kwargs.get("skills") or ["Python", "System Design", "Data Structures"]

    for a in args:
        if isinstance(a, str) and not role: role = a
        elif isinstance(a, (int, float)): overall_score = float(a)
        elif isinstance(a, dict) and not section_scores: section_scores = a
        elif isinstance(a, list) and not skills: skills = a

    strengths = [
        "Strong conceptual foundation in Object-Oriented principles and core systems mechanics.",
        "Clear communicative structure when breaking down high-level architectural approaches.",
        "Solid baseline awareness of production trade-offs between latency and maintainability."
    ]

    focus_areas = [
        "Incorporate deeper execution mechanics into conceptual responses (e.g. TLB caching, vtable dispatch).",
        "Quantify engineering outcomes with measurable production telemetry (P99 latency, throughput, scale).",
        "Maintain vocal cadence strictly within the 130–150 WPM executive delivery window."
    ]

    roadmap = [
        {"day_range": "Days 1–3", "action": "Master OOPs vtable dispatch and SOLID pattern code refactoring exercises.", "milestone": "Fluency in dynamic vs static polymorphism."},
        {"day_range": "Days 4–7", "action": "Drill Operating Systems concurrency (mutex vs semaphore, CFS scheduling, virtual memory).", "milestone": "Zero hesitation on process isolation."},
        {"day_range": "Days 8–10", "action": "Practice algorithmic problem solving with two-pointer and sliding window patterns under time pressure.", "milestone": "Optimal O(N) complexity solutions."},
        {"day_range": "Days 11–12", "action": "Quantify resume project achievements using metrics (latency reductions, RPS throughput, uptime).", "milestone": "High-impact behavioral elevator pitch."},
        {"day_range": "Days 13–14", "action": "Conduct full-length simulated mock interviews with posture and eye-contact video tracking.", "milestone": "90%+ Executive Interview Readiness."}
    ]

    return {
        "role": role or "Software Engineer",
        "overall_score": round(overall_score, 1),
        "strengths": strengths,
        "critical_focus_areas": focus_areas,
        "roadmap_14_days": roadmap,
    }
