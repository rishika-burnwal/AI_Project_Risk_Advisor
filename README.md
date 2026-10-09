# ProjectIQ — AI Project Intelligence & Risk Advisor

ProjectIQ is a Streamlit-based project intelligence workspace designed to analyze project documentation and help teams understand scope, risks, blockers, and project health. It uses retrieval-augmented generation (RAG) to retrieve relevant information from uploaded documents.

## Features

- **Project workspaces:** Create and switch between projects.
- **Document ingestion:** Upload PDF, DOCX, CSV, and TXT files.
- **Vector search:** Store document chunks and embeddings in ChromaDB.
- **AI project agents:** Scope, Risk, and Blocker agents analyze project information.
- **Project health:** Calculate a health score and review risk indicators.
- **Documentation generation:** Generate supported artifacts such as user stories, a risk register, and action items.
- **Add more documents:** Add files to an existing project's knowledge base.
- **Dark UI:** Dashboard styling with blue and indigo accents.

> Feature availability depends on the modules and integrations enabled in your local project copy.

## Tech Stack

- Python
- Streamlit
- ChromaDB
- Embedding generation module/model
- Groq API, if configured
- Document parsing libraries for supported file formats

## Workflow

1. Create a project workspace.
2. Upload project documents.
3. Extract text and split it into chunks.
4. Generate embeddings.
5. Store chunks and metadata in ChromaDB with a project identifier.
6. Run retrieval and AI analysis for the selected project.
7. Add documents as the project evolves.

## Prerequisites

- Python 3.10 or later recommended
- `pip`
- A Groq API key if your app uses Groq-powered generation

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

Replace the placeholders with your repository URL and folder name.

### 2. Create and activate a virtual environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository includes `requirements.txt`:

```bash
pip install -r requirements.txt
```

If it does not, install the packages used by your code and create a `requirements.txt`. Typical packages for this stack may include:

```bash
pip install streamlit chromadb python-dotenv groq sentence-transformers pypdf python-docx pandas
```

The exact dependencies depend on your loaders, embedding implementation, and AI provider.

### 4. Configure secrets

If the app uses Groq, create a `.env` file in the project root (do not commit it):

```env
GROQ_API_KEY=your_groq_api_key_here
```

Alternatively, use Streamlit secrets in `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Use the configuration method already supported by your code. Never publish API keys, OAuth credentials, passwords, or private documents.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, commonly `http://localhost:8501`.

## Suggested Repository Structure

Your actual folders may differ. Keep the structure required by your imports.

```text
ProjectIQ/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── rag/
│   ├── vector_store.py
│   ├── chunking.py
│   └── embeddings.py
├── ingestion/
│   └── loader.py
├── agents/
│   ├── scope_agent.py
│   ├── risk_agent.py
│   └── blocker_agent.py
└── data/
    └── chroma_db/       # generated local data; do not commit
```

## Example Use Case

For a Smart Library System, you might upload:
- Project proposal
- Software Requirements Specification (SRS)
- Meeting notes
- Task list
- Progress updates

ProjectIQ can use indexed documents as context for supported retrieval and analysis features.

## Data and Privacy

- ChromaDB persists local data under the path configured in your code (for example, `data/chroma_db`).
- Back up the database before making structural changes.
- Ensure queries filter by the active project ID so different projects do not share retrieved documents.
- Use sanitized or test documents for demonstrations if files contain confidential information.
- Keep `.env`, `.streamlit/secrets.toml`, credentials, and generated database files out of source control.

## Troubleshooting

**The app does not start**
- Activate the virtual environment.
- Install the dependencies.
- Run `streamlit run app.py` from the project root.
- Review the terminal traceback.

**A document fails to process**
- Confirm it is PDF, DOCX, CSV, or TXT.
- Check that the relevant parser library is installed.
- Review terminal logs for the loader error.

**ChromaDB errors**
- Confirm your ChromaDB version matches the API calls in the code.
- Back up the local database before changing collections.
- Avoid clearing a shared collection when switching projects.

**AI responses are empty or irrelevant**
- Confirm documents were processed and chunks stored.
- Confirm retrieval filters use the active project ID.
- Check embedding generation and API configuration.

## GitHub Safety Checklist

Add or update `.gitignore` to exclude secrets, virtual environments, and generated data. For example:

```gitignore
.venv/
venv/
__pycache__/
*.py[cod]
.env
.streamlit/secrets.toml
data/chroma_db/
*.db
credentials.json
token.json
```

Review the list to avoid publishing sensitive information or excluding files required by your deployment.

## Possible Future Improvements

- Persistent project registry across sessions
- Authentication and role-based permissions
- Project-specific audit history
- Automated ingestion, retrieval, and project-isolation tests
- Deployment configuration and monitoring

## License

No license has been specified yet. Add a `LICENSE` file before presenting the repository as open source.
