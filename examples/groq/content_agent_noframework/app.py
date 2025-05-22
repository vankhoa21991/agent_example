import os
import json
import logging
import gradio as gr
from typing import Dict, Any, Tuple

from main import ContentAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize content agent
content_agent = ContentAgent()

def process_request(topic: str, platform: str) -> Tuple[str, str, str]:
    """
    Process the content generation request.
    
    Args:
        topic: The topic to generate content for
        platform: The platform to generate content for
        
    Returns:
        Tuple: (status message, preview HTML content, JSON content)
    """
    # Generate content
    logger.info(f"Processing request for topic: {topic}, platform: {platform}")
    
    # Show status to user
    result = {
        'status': 'processing',
        'message': 'Processing your request...',
        'content': None
    }
    
    try:
        # Generate content
        response = content_agent.generate_content(topic, platform)
        
        # Process response
        if response.status == "success" and response.content:
            # Extract blog content
            blog_content = response.content.get("blog_contents", {})
            html_content = response.content.get("html_content", "")
            
            result = {
                'status': 'success',
                'message': 'Content generated successfully.',
                'content': html_content
            }
            
            # Add note about scheduling limitation
            result['message'] += "\nNote: Content scheduling is currently disabled in this demo version."
            
        else:
            result = {
                'status': 'error',
                'message': response.message or "Error generating content.",
                'content': None
            }
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        result = {
            'status': 'error',
            'message': f"Error: {str(e)}",
            'content': None
        }
    
    # Convert markdown to HTML
    if result['content']:
        content = str(result['content'])
        
        # Replace markdown headers with HTML tags
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('# '):
                lines[i] = f'<h1>{line[2:].strip()}</h1>'
            elif line.startswith('## '):
                lines[i] = f'<h2>{line[3:].strip()}</h2>'
            elif line.startswith('### '):
                lines[i] = f'<h3>{line[4:].strip()}</h3>'
        content = '\n'.join(lines)
        
        # Replace markdown bold with HTML strong tags
        content = content.replace('**', '<strong>', 1)
        while '**' in content:
            content = content.replace('**', '</strong>', 1)
            if '**' in content:
                content = content.replace('**', '<strong>', 1)
        
        result['content'] = content
    
    # Prepare response
    status = f"Status: {result['status']}\n{result['message']}"
    preview_html = result['content'] if result['content'] else ""
    content_json = json.dumps(result, indent=2) if result else ""
    
    return status, preview_html, content_json

def create_interface():
    """Create and configure the Gradio interface."""
    
    # Custom CSS
    css = """
    #preview-html {
        min-height: 200px;
        height: auto;
        border: none;
        border-radius: 10px;
        background: white;
        padding: 20px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    #preview-html strong {
        font-weight: 600;
    }
    #preview-html h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1.5rem;
        color: #1a202c;
    }
    #preview-html h2 {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.75rem 0 1.25rem;
        color: #2d3748;
    }
    #preview-html h3 {
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem;
        color: #2d3748;
    }
    .gradio-container {
        max-width: 1200px !important;
    }
    """
    
    # Create interface
    with gr.Blocks(css=css) as demo:
        gr.Markdown("""
        # Content Generator
        
        Generate content for different platforms using Groq AI models and web research.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                topic_input = gr.Textbox(
                    label="Topic",
                    placeholder="Enter a topic for content creation",
                    lines=1
                )
                
                platform_dropdown = gr.Dropdown(
                    label="Platform",
                    choices=["blog", "linkedin", "twitter", "facebook"],
                    value="blog"
                )
                
                generate_button = gr.Button("Generate Content", variant="primary")
                
                # Status output
                status_output = gr.Textbox(
                    label="Status",
                    lines=3,
                    placeholder="Generation status will appear here"
                )
            
            with gr.Column(scale=2):
                # Preview components
                gr.Markdown("### Content Preview")
                
                preview_html = gr.HTML(
                    label="Preview",
                    elem_id="preview-html"
                )
                
                # JSON output (hidden by default)
                with gr.Accordion("Raw JSON Output", open=False):
                    json_output = gr.JSON(
                        label="Content JSON"
                    )
        
        # Set up event handler for generate button
        generate_button.click(
            fn=process_request,
            inputs=[topic_input, platform_dropdown],
            outputs=[status_output, preview_html, json_output],
            api_name="generate"
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["Artificial Intelligence", "blog"],
                ["Climate Change", "linkedin"],
                ["Digital Marketing Trends", "twitter"],
                ["Remote Work", "facebook"],
            ],
            inputs=[topic_input, platform_dropdown]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Make accessible from other machines
        share=True,             # Create a public link
        inbrowser=False         # Don't try to open in browser automatically
    ) 