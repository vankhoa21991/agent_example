"""
Gradio Frontend for Content Planning Application

This module provides a user-friendly interface for generating and previewing social media content.
"""

import gradio as gr
import random
from pathlib import Path
import tempfile
import json
import logging
from enum import Enum
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the workflow
from main import ContentPlanningWorkflow

# Example topics
EXAMPLE_TOPICS = [
    "The Future of AI in Healthcare",
    "Sustainable Living in 2024",
    "The Impact of Climate Change",
    "Understanding Quantum Computing",
    "The Evolution of Social Media",
    "Mental Health in the Digital Age",
    "The Role of Blockchain",
    "Remote Work Best Practices",
]

class PostType(str, Enum):
    TWITTER = "TWITTER"
    LINKEDIN = "LINKEDIN"

def save_preview_html(html_content: str) -> str:
    """
    Save the HTML preview to a temporary file.
    
    Args:
        html_content: The HTML content to save
        
    Returns:
        str: Path to the saved HTML file
    """
    try:
        # Create a 'temp' directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Save the HTML file
        html_file = temp_dir / "preview.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return str(html_file)
    except Exception as e:
        logger.error(f"Error saving preview HTML: {str(e)}")
        return ""

def generate_content(topic: str, platform: str) -> dict:
    """
    Generate content using the workflow.
    
    Args:
        topic: The topic to generate content for
        platform: The platform to generate content for (Twitter/LinkedIn)
        
    Returns:
        dict: Generated content and status
    """
    try:
        # Create workflow instance
        url_safe_topic = topic.lower().replace(" ", "-")
        workflow = ContentPlanningWorkflow(
            session_id=f"content-plan-{url_safe_topic}",
            debug_mode=True,
        )
        
        # Run workflow
        response = workflow.run(
            topic=topic,
            post_type=PostType[platform.upper()]
        )
        
        if response and hasattr(response, 'content'):
            # Add the status note to the HTML content
            html_content = str(response.content)
            
            # Insert the status note before the closing body tag
            status_note = """
                <div style="
                    margin: 20px auto;
                    max-width: 600px;
                    padding: 10px;
                    background: #fff3cd;
                    border: 1px solid #ffeeba;
                    border-radius: 5px;
                    color: #856404;
                ">
                    <strong>Note:</strong> Content has been generated successfully but scheduling is currently disabled. 
                    You can copy the content from the JSON view and schedule it manually.
                </div>
            """
            html_content = html_content.replace('</body>', f'{status_note}</body>')
            
            return {
                "status": "success",
                "message": f"Generated {platform} content for topic: {topic}\nNote: Content scheduling is currently disabled.",
                "content": html_content,
                "preview_path": None
            }
        else:
            return {
                "status": "error",
                "message": "No content was generated. Please try again with a different topic.",
                "content": None,
                "preview_path": None
            }
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error generating content: {error_msg}")
        return {
            "status": "error",
            "message": f"Error: {error_msg}\nPlease try again or choose a different topic.",
            "content": None,
            "preview_path": None
        }

def process_request(topic: str, platform: str) -> tuple:
    """
    Process the content generation request.
    
    Args:
        topic: The topic to generate content for
        platform: The platform to generate content for
        
    Returns:
        tuple: (status message, preview HTML path, JSON content)
    """
    # Generate content
    result = generate_content(topic, platform)
    
    # Prepare response
    status = f"Status: {result['status']}\n{result['message']}"
    preview_html = result['content'] if result['content'] else ""  # Use content directly
    content_json = json.dumps(result['content'], indent=2) if result['content'] else ""
    
    return status, preview_html, content_json

def create_interface():
    """Create and configure the Gradio interface."""
    
    # Custom CSS
    css = """
    #preview-html {
        height: 600px;
        border: none;
        border-radius: 10px;
        background: white;
        overflow-y: auto;
    }
    .gradio-container {
        max-width: 1200px !important;
    }
    """
    
    # Create the interface
    with gr.Blocks(css=css) as interface:
        gr.Markdown("# ✨ Content Planning Assistant")
        gr.Markdown("Generate engaging social media content with AI assistance.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                topic_dropdown = gr.Dropdown(
                    choices=EXAMPLE_TOPICS,
                    label="Example Topics",
                    value=EXAMPLE_TOPICS[0]
                )
                custom_topic = gr.Textbox(
                    label="Or enter your own topic",
                    placeholder="e.g., The Future of AI"
                )
                platform = gr.Radio(
                    choices=["Twitter", "LinkedIn"],
                    label="Platform",
                    value="LinkedIn"
                )
                generate_btn = gr.Button("Generate Content", variant="primary")
                
                # Status output
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
                
                # JSON output
                json_output = gr.JSON(
                    label="Generated Content"
                )
            
            with gr.Column(scale=2):
                # Preview with iframe for proper rendering
                preview_html = gr.HTML(
                    label="Content Preview",
                    elem_id="preview-html"
                )
        
        def update_topic(dropdown_value, custom_value):
            """Update the topic based on dropdown or custom input."""
            return custom_value if custom_value else dropdown_value

        def get_active_topic(dropdown_value, custom_value, platform):
            """Get the active topic from either custom input or dropdown."""
            topic = custom_value if custom_value.strip() else dropdown_value
            return process_request(topic, platform)
        
        # Wire up the components
        topic_dropdown.change(
            update_topic,
            [topic_dropdown, custom_topic],
            [custom_topic]
        )
        
        generate_btn.click(
            get_active_topic,
            inputs=[topic_dropdown, custom_topic, platform],
            outputs=[status_output, preview_html, json_output]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Make accessible from other machines
        share=True,  # Create a public link
        inbrowser=True  # Open in browser automatically
    ) 