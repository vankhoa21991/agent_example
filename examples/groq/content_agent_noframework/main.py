import os
import logging
import json
from typing import Dict, Any
from dotenv import load_dotenv

from research import ResearchTeam
from writer import ContentWriter
from preview import PreviewGenerator
from models import BlogContent, RunResponse, create_error_response, create_success_response

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ContentAgent:
    """Main content agent workflow without agno dependencies"""
    
    def __init__(self):
        # Initialize components
        self.research_team = ResearchTeam()
        self.content_writer = ContentWriter()
        self.preview_generator = PreviewGenerator()
    
    def generate_content(self, topic: str, platform: str) -> RunResponse:
        """Generate content for a specific topic and platform"""
        try:
            logger.info(f"Generating content for topic: {topic}, platform: {platform}")
            
            # Step 1: Research the topic
            logger.info("Starting research...")
            research_response = self.research_team.run(topic)
            
            if research_response.status == "error":
                logger.error(f"Research failed: {research_response.message}")
                return research_response
            
            logger.info("Research completed successfully")
            
            # Step 2: Generate content based on research
            logger.info("Generating content...")
            content_response = self.content_writer.run(topic, platform, research_response)
            
            if content_response.status == "error":
                logger.error(f"Content generation failed: {content_response.message}")
                return content_response
            
            logger.info("Content generated successfully")
            
            # Step 3: Create preview and prepare response
            if not content_response.content or "blog_contents" not in content_response.content:
                return create_error_response("No blog content returned from content writer")
            
            # Extract blog content
            blog_content_dict = content_response.content["blog_contents"]
            blog_content = BlogContent(
                title=blog_content_dict["title"],
                content=blog_content_dict["content"],
                platform=blog_content_dict["platform"],
                topic=blog_content_dict["topic"],
                sources=blog_content_dict.get("sources", [])
            )
            
            # Generate preview
            logger.info("Generating preview...")
            preview_path = self.preview_generator.save_preview_html(blog_content)
            
            # Convert markdown headers to HTML for display
            content = blog_content.content
            
            # Return success response with content and preview
            response_content = {
                "blog_contents": blog_content_dict,
                "preview_path": preview_path,
                "html_content": self.preview_generator.generate_preview_html(blog_content)
            }
            
            logger.info("Content generation workflow completed successfully")
            return create_success_response(response_content)
            
        except Exception as e:
            logger.error(f"Content generation error: {str(e)}")
            return create_error_response(f"Content generation error: {str(e)}")
    
    def schedule_and_publish(self, blog_content: BlogContent, schedule_time: str = None) -> RunResponse:
        """Schedule and publish content (placeholder implementation)"""
        try:
            logger.info(f"Scheduling content: {blog_content.title} for {schedule_time if schedule_time else 'immediate'} publishing")
            
            # This is a placeholder for actual scheduling and publishing
            # In a real implementation, this would connect to various social media APIs
            
            response_content = {
                "status": "scheduled" if schedule_time else "published",
                "platform": blog_content.platform,
                "schedule_time": schedule_time,
                "title": blog_content.title
            }
            
            return create_success_response(
                response_content,
                f"Content {'scheduled' if schedule_time else 'published'} successfully"
            )
            
        except Exception as e:
            logger.error(f"Publishing error: {str(e)}")
            return create_error_response(f"Publishing error: {str(e)}")

# Simple command-line interface for testing
def cli():
    """Command-line interface for testing the content agent"""
    print("Content Agent CLI")
    print("----------------")
    
    # Initialize content agent
    agent = ContentAgent()
    
    # Get user input
    topic = input("Enter a topic: ")
    print("\nAvailable platforms:")
    print("1. LinkedIn")
    print("2. Twitter")
    print("3. Facebook")
    print("4. Blog")
    
    platform_choice = input("Choose a platform (1-4): ")
    platform_map = {
        "1": "linkedin",
        "2": "twitter",
        "3": "facebook",
        "4": "blog"
    }
    
    platform = platform_map.get(platform_choice, "blog")
    
    print(f"\nGenerating content for '{topic}' on {platform}...")
    
    # Generate content
    response = agent.generate_content(topic, platform)
    
    # Print results
    if response.status == "success" and response.content:
        blog_content = response.content.get("blog_contents", {})
        preview_path = response.content.get("preview_path", "")
        
        print("\nContent generated successfully!")
        print(f"Title: {blog_content.get('title')}")
        print(f"Preview saved to: {preview_path}")
        
        # Ask to open preview
        if preview_path and os.path.exists(preview_path):
            open_preview = input("\nOpen preview? (y/n): ")
            if open_preview.lower() == "y":
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(preview_path)}")
    else:
        print(f"\nError: {response.message}")

if __name__ == "__main__":
    cli() 