import os
import logging
from dataclasses import asdict
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

from models import BlogContent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HTML templates for different platforms
TEMPLATES = {
    "linkedin": """
    <div class="linkedin-post">
        <div class="post-header">
            <img src="https://via.placeholder.com/60" alt="Profile Picture" class="profile-pic">
            <div class="post-meta">
                <h3>Your Name</h3>
                <p class="post-info">Content Creator • Just now</p>
            </div>
        </div>
        
        <div class="post-content">
            {{ content|safe }}
        </div>
        
        <div class="post-actions">
            <div class="action"><span class="icon">👍</span> Like</div>
            <div class="action"><span class="icon">💬</span> Comment</div>
            <div class="action"><span class="icon">↗️</span> Share</div>
        </div>
    </div>
    """,
    
    "twitter": """
    <div class="twitter-post">
        <div class="post-header">
            <img src="https://via.placeholder.com/50" alt="Profile Picture" class="profile-pic">
            <div class="post-meta">
                <h3>Your Name <span class="handle">@yourhandle</span></h3>
            </div>
        </div>
        
        <div class="post-content">
            {{ content|safe }}
        </div>
        
        <div class="post-info">
            <span>Just now • Twitter Web App</span>
        </div>
        
        <div class="post-actions">
            <div class="action"><span class="icon">💬</span> Reply</div>
            <div class="action"><span class="icon">🔄</span> Retweet</div>
            <div class="action"><span class="icon">❤️</span> Like</div>
            <div class="action"><span class="icon">📤</span> Share</div>
        </div>
    </div>
    """,
    
    "facebook": """
    <div class="facebook-post">
        <div class="post-header">
            <img src="https://via.placeholder.com/50" alt="Profile Picture" class="profile-pic">
            <div class="post-meta">
                <h3>Your Name</h3>
                <p class="post-info">Just now • <span class="icon">🌎</span></p>
            </div>
        </div>
        
        <div class="post-content">
            {{ content|safe }}
        </div>
        
        <div class="post-actions">
            <div class="action"><span class="icon">👍</span> Like</div>
            <div class="action"><span class="icon">💬</span> Comment</div>
            <div class="action"><span class="icon">↗️</span> Share</div>
        </div>
    </div>
    """,
    
    "blog": """
    <div class="blog-post">
        <h1 class="blog-title">{{ title }}</h1>
        <div class="post-meta">
            <p>Published just now • <span class="reading-time">{{ reading_time }} min read</span></p>
        </div>
        
        <div class="blog-content">
            {{ content|safe }}
        </div>
        
        <div class="sources-section">
            <h3>Sources</h3>
            <ul class="sources-list">
                {% for source in sources %}
                <li>
                    {% if source.source_type == "wikipedia" %}📚{% else %}🌐{% endif %}
                    <a href="{{ source.url }}" target="_blank">{{ source.title }}</a>
                    <span class="source-name">{{ source.source }}</span>
                </li>
                {% endfor %}
            </ul>
        </div>
    </div>
    """
}

# Base CSS styles
BASE_CSS = """
* {
    box-sizing: border-box;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

body {
    background-color: #f5f5f5;
    margin: 0;
    padding: 20px;
}

.container {
    max-width: 600px;
    margin: 0 auto;
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.profile-pic {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    object-fit: cover;
}

.post-header {
    display: flex;
    align-items: center;
    padding: 16px;
    border-bottom: 1px solid #f0f0f0;
}

.post-meta {
    margin-left: 12px;
}

.post-meta h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
}

.post-info {
    margin: 0;
    font-size: 14px;
    color: #65676B;
}

.post-content {
    padding: 16px;
    font-size: 15px;
    line-height: 1.5;
    color: #1c1e21;
}

.post-actions {
    display: flex;
    justify-content: space-around;
    padding: 8px 16px;
    border-top: 1px solid #f0f0f0;
}

.action {
    display: flex;
    align-items: center;
    font-size: 14px;
    color: #65676B;
    cursor: pointer;
}

.action:hover {
    color: #1877F2;
}

.icon {
    margin-right: 6px;
}

/* Blog specific styles */
.blog-post {
    padding: 24px;
}

.blog-title {
    font-size: 28px;
    margin-top: 0;
    margin-bottom: 16px;
    color: #333;
}

.reading-time {
    color: #65676B;
}

.blog-content {
    font-size: 16px;
    line-height: 1.6;
    color: #333;
    margin-bottom: 32px;
}

.blog-content h3 {
    font-size: 20px;
    margin-top: 28px;
    margin-bottom: 16px;
    color: #333;
}

.sources-section {
    border-top: 1px solid #f0f0f0;
    padding-top: 20px;
    margin-top: 32px;
}

.sources-list {
    padding-left: 0;
    list-style-type: none;
}

.sources-list li {
    margin-bottom: 8px;
    font-size: 14px;
}

.sources-list a {
    color: #1877F2;
    text-decoration: none;
}

.sources-list a:hover {
    text-decoration: underline;
}

.source-name {
    color: #65676B;
    font-size: 13px;
    margin-left: 6px;
}

/* Twitter specific */
.twitter-post {
    background-color: white;
}

.twitter-post .handle {
    color: #657786;
    font-weight: normal;
    font-size: 14px;
}

/* LinkedIn specific */
.linkedin-post {
    background-color: white;
}

/* Facebook specific */
.facebook-post {
    background-color: white;
}
"""

class PreviewGenerator:
    """Generate HTML preview for content without agno dependencies"""
    
    def __init__(self):
        # Create temp directory if it doesn't exist
        self.temp_dir = Path(__file__).parent / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment with templates
        self.templates = {}
        for platform, template_str in TEMPLATES.items():
            self.templates[platform] = Environment(loader=FileSystemLoader(self.temp_dir)).from_string(template_str)
    
    def estimate_reading_time(self, content: str) -> int:
        """Estimate reading time in minutes"""
        # Average reading speed: 200-250 words per minute
        word_count = len(content.split())
        reading_time = max(1, round(word_count / 225))
        return reading_time
    
    def generate_preview_html(self, blog_content: BlogContent) -> str:
        """Generate HTML preview for the given blog content"""
        try:
            platform = blog_content.platform.lower()
            template = self.templates.get(platform)
            
            if not template:
                logger.warning(f"No template found for platform: {platform}. Using blog template.")
                template = self.templates.get("blog")
            
            reading_time = self.estimate_reading_time(blog_content.content)
            
            # Render template
            html_content = template.render(
                title=blog_content.title,
                content=blog_content.content,
                sources=blog_content.sources,
                reading_time=reading_time
            )
            
            # Add container and CSS
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Content Preview</title>
                <style>
                {BASE_CSS}
                </style>
            </head>
            <body>
                <div class="container">
                    {html_content}
                </div>
            </body>
            </html>
            """
            
            return full_html
            
        except Exception as e:
            logger.error(f"Error generating preview HTML: {str(e)}")
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Preview Error</title>
            </head>
            <body>
                <h1>Error Generating Preview</h1>
                <p>{str(e)}</p>
            </body>
            </html>
            """
    
    def save_preview_html(self, blog_content: BlogContent) -> str:
        """Save HTML preview to a file and return the file path"""
        try:
            html_content = self.generate_preview_html(blog_content)
            
            # Create a filename based on topic and platform
            safe_topic = "".join(c if c.isalnum() else "_" for c in blog_content.topic)
            filename = f"{safe_topic}_{blog_content.platform}.html"
            file_path = self.temp_dir / filename
            
            # Save HTML to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving preview HTML: {str(e)}")
            return ""

# Simple test function
def test_preview():
    from models import NewsArticle, BlogContent
    
    # Create sample blog content
    blog_content = BlogContent(
        title="The Future of AI in Healthcare",
        content="""
        <h3>Revolutionizing Patient Care</h3>
        <p>Artificial intelligence is transforming healthcare in unprecedented ways. From <strong>early disease detection</strong> to personalized treatment plans, AI technologies are enabling healthcare providers to deliver more effective and efficient care.</p>
        
        <h3>Diagnostic Innovations</h3>
        <p>AI algorithms can now analyze medical images with remarkable accuracy. Studies show that AI can detect certain conditions with <strong>97% accuracy</strong>, sometimes exceeding human capabilities.</p>
        
        <h3>Challenges and Ethical Considerations</h3>
        <p>Despite these advances, the healthcare industry faces significant challenges in AI adoption, including data privacy concerns and the need for regulatory frameworks that ensure patient safety.</p>
        """,
        platform="blog",
        topic="AI in Healthcare",
        sources=[
            NewsArticle(
                title="AI Revolution in Healthcare",
                content="Overview of AI applications in medical field",
                url="https://example.com/article1",
                source="Medical Journal",
                source_type="web"
            ),
            NewsArticle(
                title="Artificial Intelligence",
                content="Wikipedia article on AI applications",
                url="https://en.wikipedia.org/wiki/Artificial_intelligence",
                source="Wikipedia",
                source_type="wikipedia"
            )
        ]
    )
    
    # Generate preview
    preview_generator = PreviewGenerator()
    preview_path = preview_generator.save_preview_html(blog_content)
    
    print(f"Preview saved to: {preview_path}")

if __name__ == "__main__":
    test_preview() 