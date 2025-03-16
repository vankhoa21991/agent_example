import os
from typing import List, Dict, Any
from pathlib import Path
import tempfile
import requests
import json
from datetime import datetime
from urllib.parse import urlparse

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# PDF processing
import PyPDF2

# Required for web scraping
from bs4 import BeautifulSoup
import httpx

# Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-api-key")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize Groq LLM
groq_llm = ChatGroq(
    model_name="llama3-70b-8192",
    api_key=GROQ_API_KEY
)

# Define the agent state
from typing_extensions import TypedDict

class AgentState(TypedDict):
    pdf_paths: List[str]
    urls: List[str]
    pdf_contents: Dict[str, str]
    url_contents: Dict[str, str]
    final_summary: str
    plan: List[str]
    status: str
    errors: List[str]
    output_file: str

# Define the agent steps
def create_plan(state: AgentState) -> AgentState:
    """Create a plan for processing the documents"""
    print("\n=== CREATING PLAN ===")
    print(f"Processing {len(state['pdf_paths'])} PDFs and {len(state['urls'])} URLs")
    
    prompt = f"""
    You are an agent tasked with planning the processing of multiple documents.
    
    Here are the inputs:
    - PDF paths: {state["pdf_paths"]}
    - URLs: {state["urls"]}
    
    Create a step-by-step plan for processing these documents to generate a comprehensive summary.
    """
    
    print("Sending prompt to Groq LLM...")
    response = groq_llm.invoke([{"role": "user", "content": prompt}])
    # Extract the content from the response
    plan_text = response.content
    print("\nGroq LLM Response:")
    print(plan_text)
    
    # Extract the plan steps
    plan_lines = [line.strip() for line in plan_text.split("\n") if line.strip()]
    plan_steps = [line for line in plan_lines if line.startswith("- ") or line.startswith("1. ")]
    
    # Remove the prefix from each step
    plan_steps = [step[2:].strip() if step.startswith("- ") else step[3:].strip() for step in plan_steps]
    
    # Create a new state with updated values
    return {
        **state,
        "plan": plan_steps,
        "status": "processing_pdfs"
    }

def process_pdfs(state: AgentState) -> AgentState:
    """Process all PDF files using PyPDF2"""
    print("\n=== PROCESSING PDFs ===")
    pdf_contents = state["pdf_contents"].copy()
    errors = state["errors"].copy()
    
    for pdf_path in state["pdf_paths"]:
        print(f"Processing PDF: {os.path.basename(pdf_path)}")
        try:
            # Load the PDF using PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                print(f"  - PDF has {len(reader.pages)} pages")
                
                # Extract text content from all pages
                text_content = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text_content += page.extract_text() + "\n\n"
            
            # Store the content
            pdf_contents[pdf_path] = text_content
            print(f"  - Successfully extracted {len(text_content)} characters")
        except Exception as e:
            print(f"  - ERROR: {str(e)}")
            errors.append(f"Error processing PDF {pdf_path}: {str(e)}")
    
    # Create a new state with updated values
    return {
        **state,
        "pdf_contents": pdf_contents,
        "errors": errors,
        "status": "processing_urls"
    }

def process_urls(state: AgentState) -> AgentState:
    """Process all URLs by scraping their content"""
    print("\n=== PROCESSING URLs ===")
    url_contents = state["url_contents"].copy()
    errors = state["errors"].copy()
    
    for url in state["urls"]:
        print(f"Processing URL: {url}")
        try:
            # Get the webpage content
            print(f"  - Fetching content...")
            response = httpx.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML
            print(f"  - Parsing HTML...")
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract text content (prioritize article content)
            article = soup.find('article')
            if article:
                print(f"  - Found article element, extracting text...")
                text_content = article.get_text(separator="\n\n")
            else:
                print(f"  - No article element found, extracting from full page...")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Get text
                text_content = soup.get_text(separator="\n\n")
                
                # Clean up whitespace
                lines = (line.strip() for line in text_content.split("\n"))
                text_content = "\n\n".join(line for line in lines if line)
            
            # Store the content
            url_contents[url] = text_content
            print(f"  - Successfully extracted {len(text_content)} characters")
        except Exception as e:
            print(f"  - ERROR: {str(e)}")
            errors.append(f"Error processing URL {url}: {str(e)}")
    
    # Create a new state with updated values
    return {
        **state,
        "url_contents": url_contents,
        "errors": errors,
        "status": "summarizing"
    }

def generate_summary(state: AgentState) -> AgentState:
    """Generate a summary of all processed content"""
    print("\n=== GENERATING SUMMARY ===")
    
    # Combine all content from PDFs and URLs
    all_content = ""
    
    # Add PDF content
    print("Preparing content for summarization...")
    for pdf_path, content in state["pdf_contents"].items():
        filename = os.path.basename(pdf_path)
        print(f"  - Including content from PDF: {filename} ({len(content)} characters)")
        all_content += f"## PDF: {filename}\n\n{content[:2000]}...\n\n"
    
    # Add URL content
    for url, content in state["url_contents"].items():
        domain = urlparse(url).netloc
        print(f"  - Including content from URL: {domain} ({len(content)} characters)")
        all_content += f"## URL: {domain}\n\n{content[:2000]}...\n\n"
    
    # Generate the summary
    print("Sending content to Groq LLM for summarization...")
    prompt = f"""
    You are a comprehensive summarizer. Analyze the following content from multiple PDFs and websites,
    and create a cohesive summary that integrates the key information from all sources.
    
    Here is the content to summarize:
    
    {all_content}
    
    Create a well-structured summary that:
    1. Identifies the main topics across all documents
    2. Highlights key findings and insights
    3. Organizes the information in a logical flow
    4. Provides a conclusion that ties everything together
    
    Format your response in Markdown with appropriate headings, bullet points, and formatting.
    """
    
    response = groq_llm.invoke([{"role": "user", "content": prompt}])
    # Extract the content from the response
    summary_text = response.content
    print("\nSummary generated successfully!")
    print(f"Summary length: {len(summary_text)} characters")
    
    # Create a new state with updated values
    return {
        **state,
        "final_summary": summary_text,
        "status": "complete"
    }

def save_output(state: AgentState) -> AgentState:
    """Save the final summary as a markdown file"""
    print("\n=== SAVING OUTPUT ===")
    
    # Create output filename based on current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"summary_{timestamp}.md"
    print(f"Creating output file: {output_file}")
    
    # Create the markdown content
    print("Formatting markdown content...")
    markdown_content = f"""# Document Processing Summary
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Sources Processed
### PDFs:
{chr(10).join('- ' + os.path.basename(pdf) for pdf in state["pdf_paths"])}

### URLs:
{chr(10).join('- ' + url for url in state["urls"])}

## Summary
{state["final_summary"]}

## Processing Notes
{chr(10).join('- ' + error for error in state["errors"]) if state["errors"] else "No errors during processing."}
"""
    
    # Write the markdown file
    print(f"Writing {len(markdown_content)} characters to file...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"File saved successfully!")
    
    # Create a new state with updated values
    return {
        **state,
        "output_file": str(output_file)
    }

def route_next_step(state: AgentState) -> str:
    """Determine the next step based on the current status"""
    print("\n=== ROUTING NEXT STEP ===")
    print(f"Current status: {state['status']}")
    
    next_step = ""
    if state["status"] == "planning":
        next_step = "create_plan"
    elif state["status"] == "processing_pdfs":
        next_step = "process_pdfs"
    elif state["status"] == "processing_urls":
        next_step = "process_urls"
    elif state["status"] == "summarizing":
        next_step = "generate_summary"
    elif state["status"] == "complete":
        next_step = "save_output"
    else:
        next_step = END
    
    print(f"Next step: {next_step}")
    return next_step

# Build the agent graph
def build_agent_graph() -> StateGraph:
    """Build the LangGraph agent for document processing"""
    # Initialize the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("create_plan", create_plan)
    graph.add_node("process_pdfs", process_pdfs)
    graph.add_node("process_urls", process_urls)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("save_output", save_output)
    
    # Add conditional edges
    graph.add_conditional_edges("create_plan", route_next_step)
    graph.add_conditional_edges("process_pdfs", route_next_step)
    graph.add_conditional_edges("process_urls", route_next_step)
    graph.add_conditional_edges("generate_summary", route_next_step)
    graph.add_conditional_edges("save_output", lambda _: END)
    
    # Set the entry point
    graph.set_entry_point("create_plan")
    
    return graph

# Function to print agent state
def print_agent_state(state: AgentState, title: str = "AGENT STATE"):
    """Print the current state of the agent in a readable format"""
    print(f"\n{'=' * 40}")
    print(f"{title}")
    print(f"{'=' * 40}")
    print(f"Status: {state['status']}")
    print(f"PDFs: {len(state['pdf_paths'])} files")
    print(f"URLs: {len(state['urls'])} links")
    print(f"PDF Contents: {len(state['pdf_contents'])} processed")
    print(f"URL Contents: {len(state['url_contents'])} processed")
    print(f"Plan Steps: {len(state['plan'])}")
    print(f"Errors: {len(state['errors'])}")
    if state['final_summary']:
        print(f"Summary: {len(state['final_summary'])} characters")
    if state['output_file']:
        print(f"Output File: {state['output_file']}")
    print(f"{'=' * 40}")

# Main function to run the agent
def run_document_processor(pdf_paths: List[str], urls: List[str]) -> str:
    """
    Run the document processing agent on the provided PDFs and URLs
    
    Args:
        pdf_paths: List of paths to PDF files
        urls: List of URLs to process
    
    Returns:
        Path to the generated markdown file
    """
    print("\n========================================")
    print("STARTING DOCUMENT PROCESSING AGENT")
    print("========================================")
    print(f"Processing {len(pdf_paths)} PDFs and {len(urls)} URLs")
    
    # Initialize agent state
    print("\nInitializing agent state...")
    initial_state = AgentState(
        pdf_paths=pdf_paths,
        urls=urls,
        pdf_contents={},
        url_contents={},
        final_summary="",
        plan=[],
        status="planning",
        errors=[],
        output_file=""
    )
    
    # Print initial state
    print_agent_state(initial_state, "INITIAL AGENT STATE")
    
    # Build the agent graph
    print("Building agent graph...")
    graph = build_agent_graph()
    
    # Create a checkpointer to track the state
    print("Setting up checkpointer...")
    checkpointer = MemorySaver()
    
    # Compile the graph
    print("Compiling graph...")
    app = graph.compile(checkpointer=checkpointer)
    
    # Run the agent with configuration for the checkpointer
    print("\nRunning agent workflow...")
    config = {"configurable": {"thread_id": "doc_processor"}}
    
    # Use stream to see intermediate states
    print("\nStreaming agent execution:")
    states = []
    for state in app.stream(initial_state, config=config):
        states.append(state)
        # Get the last key which is the node that just executed
        last_node = list(state.keys())[-1]
        print(f"\n{'*' * 60}")
        print(f"EXECUTED NODE: {last_node}")
        print(f"{'*' * 60}")
        print_agent_state(state[last_node], f"STATE AFTER {last_node.upper()}")
    
    # Get the final state
    final_state = states[-1][list(states[-1].keys())[-1]]
    
    print("\n========================================")
    print("DOCUMENT PROCESSING COMPLETE")
    print("========================================")
    
    return final_state["output_file"]

# Example usage
if __name__ == "__main__":
    # Sample inputs
    pdf_paths = [
        # "/home/vankhoa@median.cad/code/github/agent_example/data/Weaviate Agentic Architectures-ebook.pdf",
        # "/home/vankhoa@median.cad/code/github/agent_example/data/fibromyalgia.pdf",
    ]
    
    urls = [
        "https://github.com/oumi-ai/oumi?tab=readme-ov-file",
        "https://python.langchain.com/docs/integrations/llms/groq",
    ]
    
    # Run the document processor
    output_file = run_document_processor(pdf_paths, urls)
    print(f"Processing complete. Output saved to: {output_file}")
