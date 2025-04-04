import json
from typing import List, Optional, Union
import webbrowser
import os
from pathlib import Path
import tempfile

from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.run.response import RunEvent
from agno.tools.firecrawl import FirecrawlTools
from agno.utils.log import logger
from agno.workflow import Workflow
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from config import PostType
from prompts import (
    agents_config,
    tasks_config,
)
from scheduler import schedule

# Load environment variables
load_dotenv()


# Define Pydantic models to structure responses
class BlogAnalyzer(BaseModel):
    """
    Represents the response from the Blog Analyzer agent.
    Includes the blog title and content in Markdown format.
    """

    title: str
    blog_content_markdown: str


class Tweet(BaseModel):
    """
    Represents an individual tweet within a Twitter thread.
    """

    content: str
    is_hook: bool = Field(
        description="Marks if this tweet is the 'hook' (first tweet)"
    )
    media_urls: Optional[List[str]] = Field(
        default_factory=list, description="Associated media URLs, if any"
    )  # type: ignore


class Thread(BaseModel):
    """
    Represents a complete Twitter thread containing multiple tweets.
    """

    topic: str
    tweets: List[Tweet]


class LinkedInPost(BaseModel):
    """
    Represents a LinkedIn post.
    """

    content: str
    media_url: Optional[List[str]] = None  # Optional media attachment URL


class ContentPlanningWorkflow(Workflow):
    """
    This workflow automates the process of:
    1. Scraping a blog post using the Blog Analyzer agent.
    2. Generating a content plan for either Twitter or LinkedIn based on the scraped content.
    3. Scheduling and publishing the planned content.
    """

    # This description is used only in workflow UI
    description: str = (
        "Plan, schedule, and publish social media content based on a blog post."
    )

    # Blog Analyzer Agent: Extracts blog content (title, sections) and converts it into Markdown format for further use.
    blog_analyzer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[
            FirecrawlTools(scrape=True, crawl=False)
        ],  # Enables blog scraping capabilities
        description=f"{agents_config['blog_analyzer']['role']} - {agents_config['blog_analyzer']['goal']}",
        instructions=[
            f"{agents_config['blog_analyzer']['backstory']}",
            tasks_config["analyze_blog"][
                "description"
            ],  # Task-specific instructions for blog analysis
        ],
        response_model=BlogAnalyzer,  # Expects response to follow the BlogAnalyzer Pydantic model
    )

    # Twitter Thread Planner: Creates a Twitter thread from the blog content, each tweet is concise, engaging,
    # and logically connected with relevant media.
    twitter_thread_planner: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        description=f"{agents_config['twitter_thread_planner']['role']} - {agents_config['twitter_thread_planner']['goal']}",
        instructions=[
            f"{agents_config['twitter_thread_planner']['backstory']}",
            tasks_config["create_twitter_thread_plan"]["description"],
        ],
        response_model=Thread,  # Expects response to follow the Thread Pydantic model
    )

    # LinkedIn Post Planner: Converts blog content into a structured LinkedIn post, optimized for a professional
    # audience with relevant hashtags.
    linkedin_post_planner: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        description=f"{agents_config['linkedin_post_planner']['role']} - {agents_config['linkedin_post_planner']['goal']}",
        instructions=[
            f"{agents_config['linkedin_post_planner']['backstory']}",
            tasks_config["create_linkedin_post_plan"]["description"],
        ],
        response_model=LinkedInPost,  # Expects response to follow the LinkedInPost Pydantic model
    )

    def scrape_blog_post(self, blog_post_url: str, use_cache: bool = True):
        """
        Scrapes a blog post and returns its content in markdown format.
        
        Args:
            blog_post_url: URL of the blog post to scrape
            use_cache: Whether to use cached content if available
            
        Returns:
            str: The blog content in markdown format
            
        Raises:
            ValueError: If the blog analyzer returns unexpected content type
            Exception: If there's an error during scraping
        """
        try:
            if use_cache and blog_post_url in self.session_state:
                logger.info(f"Using cache for blog post: {blog_post_url}")
                return self.session_state[blog_post_url]
            
            response: RunResponse = self.blog_analyzer.run(blog_post_url)
            if not response or not response.content:
                raise ValueError("Empty response received from blog analyzer")
                
            if isinstance(response.content, BlogAnalyzer):
                result = response.content
                logger.info(f"Blog title: {result.title}")
                self.session_state[blog_post_url] = result.blog_content_markdown
                return result.blog_content_markdown
            else:
                raise ValueError(f"Unexpected content type received from blog analyzer: {type(response.content)}")
        except Exception as e:
            logger.error(f"Error scraping blog post: {str(e)}")
            raise

    def generate_plan(self, blog_content: str, post_type: PostType):
        """
        Generates a content plan based on the blog content and post type.
        
        Args:
            blog_content: The blog content in markdown format
            post_type: Type of post to generate (Twitter or LinkedIn)
            
        Returns:
            Union[Thread, LinkedInPost]: The generated content plan
            
        Raises:
            ValueError: If post_type is unsupported or if the planner returns unexpected content
        """
        try:
            if post_type == PostType.TWITTER:
                logger.info("Generating Twitter thread plan")
                planner = self.twitter_thread_planner
                response = planner.run(blog_content)
                if isinstance(response.content, Thread):
                    if response.content.tweets:
                        response.content.tweets[0].is_hook = True
                    return response.content
            elif post_type == PostType.LINKEDIN:
                logger.info("Generating LinkedIn post plan")
                planner = self.linkedin_post_planner
                response = planner.run(blog_content)
                if isinstance(response.content, LinkedInPost):
                    return response.content
            else:
                raise ValueError(f"Unsupported post type: {post_type}")

            # Handle string response (fallback case)
            if isinstance(response.content, str):
                try:
                    data = json.loads(response.content)
                    if post_type == PostType.TWITTER:
                        thread = Thread(**data)
                        if thread.tweets:
                            thread.tweets[0].is_hook = True
                        return thread
                    else:
                        return LinkedInPost(**data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse planner response as JSON: {str(e)}")

            raise ValueError(f"Unexpected content type received from planner: {type(response.content)}")
        except Exception as e:
            logger.error(f"Error generating content plan: {str(e)}")
            raise

    def schedule_and_publish(self, plan, post_type: PostType) -> RunResponse:
        """
        Schedules and publishes the content leveraging Typefully api.
        
        Args:
            plan: The content plan to schedule (Thread or LinkedInPost)
            post_type: Type of post to publish (Twitter or LinkedIn)
            
        Returns:
            RunResponse: Response containing the scheduling result
            
        Raises:
            ValueError: If plan is not of the expected type for the given post_type
        """
        try:
            logger.info(f"Publishing content for post type: {post_type}")
            
            # Validate plan type
            if post_type == PostType.TWITTER and not isinstance(plan, Thread):
                raise ValueError(f"Expected Thread for Twitter post, got {type(plan)}")
            elif post_type == PostType.LINKEDIN and not isinstance(plan, LinkedInPost):
                raise ValueError(f"Expected LinkedInPost for LinkedIn post, got {type(plan)}")

            # Use the `scheduler` module to schedule the content
            response = schedule(
                thread_model=plan,
                post_type=post_type,
            )

            if response:
                logger.info("Content successfully scheduled")
                return RunResponse(content=response, event=RunEvent.workflow_completed)
            else:
                logger.error("Failed to schedule content")
                return RunResponse(
                    content="Failed to schedule content.",
                    event=RunEvent.workflow_completed
                )
        except Exception as e:
            logger.error(f"Error scheduling content: {str(e)}")
            return RunResponse(
                content=f"Error scheduling content: {str(e)}",
                event=RunEvent.workflow_completed
            )

    def run(self, blog_post_url, post_type) -> RunResponse:
        """
        Args:
            blog_post_url: URL of the blog post to analyze.
            post_type: Type of post to generate (e.g., Twitter or LinkedIn).
        """
        # Scrape the blog post
        blog_content = self.scrape_blog_post(blog_post_url)

        # Generate the plan based on the blog and post type
        plan = self.generate_plan(blog_content, post_type)

        # Schedule and publish the content
        response = self.schedule_and_publish(plan, post_type)

        return response


if __name__ == "__main__":
    # Initialize and run the workflow
    blogpost_url = "https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag"
    workflow = ContentPlanningWorkflow()
    post_response = workflow.run(
        blog_post_url=blogpost_url, post_type=PostType.LINKEDIN
    )  # PostType.LINKEDIN for LinkedIn post
    
    # Create a temporary directory to store the HTML file
    temp_dir = tempfile.mkdtemp()
    html_file = Path(temp_dir) / "response.html"
    
    # Extract HTML content from the response
    if isinstance(post_response.content, dict):
        html_content = post_response.content.get('html', '')
    else:
        html_content = str(post_response.content)
    
    # Create a proper HTML document with styling
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Content Preview</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 10px 0;
            }}
            a {{
                color: #1DA1F2;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Write the HTML content to a file
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(full_html)
    
    # Open the HTML file in the default browser
    webbrowser.open(f"file://{html_file}")
    
    logger.info(f"Response saved to: {html_file}")
    logger.info("Content preview opened in browser")
