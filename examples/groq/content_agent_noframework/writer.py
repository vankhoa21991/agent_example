import os
import json
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from groq import Groq

from models import BlogContent, create_error_response, create_success_response, RunResponse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Platform-specific prompt templates
PLATFORM_PROMPTS = {
    "linkedin": """
    Create engaging LinkedIn content that is professional and thought-provoking:
    - Use a professional, insightful tone
    - Include 2-3 short paragraphs (3-4 sentences each)
    - Include 1-2 relevant statistics or facts
    - Add 3-4 relevant hashtags at the end
    - Keep the content under 1300 characters
    """,
    
    "twitter": """
    Create concise Twitter content that grabs attention:
    - Use a conversational, direct tone
    - Keep it under 280 characters
    - Include 1 key insight or fact
    - Use 1-2 relevant hashtags
    - Consider adding a question to engage readers
    """,
    
    "facebook": """
    Create Facebook content that encourages engagement:
    - Use a friendly, conversational tone
    - Include 2-3 paragraphs of varying length
    - Ask a question to encourage comments
    - Include 1-2 relevant statistics or facts
    - End with a call to action (like, comment, share)
    """,
    
    "blog": """
    Create a comprehensive blog post with these elements:
    - Attention-grabbing title
    - Introduction (2-3 paragraphs that explain the topic's importance)
    - 3-4 main sections with subheadings
    - Each section should cover a key aspect of the topic
    - Include statistics, examples, and actionable advice
    - Conclusion that summarizes key points and includes a call to action
    - Format with appropriate HTML tags: <h3> for headings, <strong> for emphasis
    """
}

class ContentWriter:
    """Handles content writing using Groq directly instead of agents"""
    
    def __init__(self):
        # Use Groq for LLM functionality
        self.model = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def generate_content(self, topic: str, platform: str, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content for a specific platform based on research data"""
        
        # Get platform-specific prompt
        platform_prompt = PLATFORM_PROMPTS.get(platform.lower(), PLATFORM_PROMPTS["blog"])
        
        # Format research data for the prompt
        articles = research_data.get("articles", [])
        analysis = research_data.get("analysis", {})
        
        insights = analysis.get("insights", [])
        facts = analysis.get("facts", [])
        perspectives = analysis.get("perspectives", [])
        summary = analysis.get("summary", "")
        
        # Format source information
        sources_text = ""
        for i, article in enumerate(articles[:5]):  # Limit to first 5 sources
            source_icon = "📚" if article.get("source_type") == "wikipedia" else "🌐"
            sources_text += f"{i+1}. {source_icon} {article.get('title', 'Untitled')} - {article.get('url', 'No URL')}\n"
        
        # Create the full prompt
        prompt = f"""
        You are a professional content writer. I need you to create {platform} content about "{topic}".
        
        {platform_prompt}
        
        Based on the research provided, here are key points to include:
        
        INSIGHTS:
        {json.dumps(insights, indent=2)}
        
        FACTS:
        {json.dumps(facts, indent=2)}
        
        PERSPECTIVES:
        {json.dumps(perspectives, indent=2)}
        
        SUMMARY:
        {summary}
        
        SOURCES:
        {sources_text}
        
        The content should be factually accurate and based on the research provided.
        For blog posts, use proper HTML formatting with <h3> for headings and <strong> for emphasis.
        
        Return ONLY the content, without any comments or explanations.
        """
        
        try:
            completion = self.model.chat.completions.create(
                model="llama3-70b-8192",  # Using Llama3-70B model
                messages=[
                    {"role": "system", "content": "You are a professional content writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2500
            )
            
            content = completion.choices[0].message.content
            
            # Generate a title
            title_prompt = f"""
            Create a compelling title for {platform} content about "{topic}".
            The content discusses:
            
            {json.dumps(insights[:2], indent=2)}
            
            Keep the title concise, engaging, and relevant to the topic.
            Return ONLY the title text, without quotes or any other text.
            """
            
            title_completion = self.model.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a professional content writer."},
                    {"role": "user", "content": title_prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            title = title_completion.choices[0].message.content.strip().replace('"', '')
            
            return {
                "title": title,
                "content": content,
                "platform": platform,
                "topic": topic
            }
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return {
                "title": f"Error: {topic}",
                "content": f"Error generating content: {str(e)}",
                "platform": platform,
                "topic": topic
            }
    
    def run(self, topic: str, platform: str, research_response: RunResponse) -> RunResponse:
        """Main method to run content generation based on research"""
        try:
            # Validate input
            if not research_response.content:
                return create_error_response("No research data provided")
            
            # Generate content
            content_data = self.generate_content(topic, platform, research_response.content)
            
            # Create blog content object
            blog_content = BlogContent(
                title=content_data["title"],
                content=content_data["content"],
                platform=platform,
                topic=topic,
                sources=[article for article in research_response.content.get("articles", [])]
            )
            
            # Format response
            response_content = {
                "blog_contents": blog_content.__dict__
            }
            
            return create_success_response(response_content)
            
        except Exception as e:
            logger.error(f"Content generation error: {str(e)}")
            return create_error_response(f"Content generation error: {str(e)}")

# Simple test function
def test_content_writer():
    from research import ResearchTeam
    
    # Run research first
    research_team = ResearchTeam()
    research_response = research_team.run("climate change innovations")
    
    # Generate content
    writer = ContentWriter()
    content_response = writer.run("climate change innovations", "blog", research_response)
    
    print(json.dumps(content_response.__dict__, indent=2))

if __name__ == "__main__":
    test_content_writer() 