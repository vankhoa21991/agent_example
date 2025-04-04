"""
Content Planning Workflow

This module orchestrates the content planning process, from research to content generation
and preview creation.
"""

import json
from typing import List, Optional, Union
import webbrowser
import os
from pathlib import Path
import tempfile

from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.run.response import RunEvent
from agno.utils.log import logger
from agno.workflow import Workflow
from dotenv import load_dotenv
from rich.console import Console

from config import PostType
from prompts import (
    agents_config,
    tasks_config,
)
from scheduler import schedule
from research_team import ResearchTeam, BlogContent
from preview import create_preview
from models import Tweet, Thread, LinkedInPost
from writing import ContentWriter

# Load environment variables
load_dotenv()


class ContentPlanningWorkflow(Workflow):
    """
    This workflow automates the process of:
    1. Using the Research Team to find and analyze content
    2. Generating a content plan for either Twitter or LinkedIn based on the analyzed content
    3. Scheduling and publishing the planned content
    """

    # This description is used only in workflow UI
    description: str = (
        "Plan, schedule, and publish social media content based on researched content."
    )

    # Research Team: Finds and analyzes relevant content
    research_team: ResearchTeam = ResearchTeam(
        session_id="research-team",
        debug_mode=True,
    )

    # Content Writer: Creates and improves social media content
    content_writer: ContentWriter = ContentWriter(
        session_id="content-writer",
        debug_mode=True,
    )

    def generate_plan(self, blog_contents: List[BlogContent], post_type: PostType):
        """
        Generates a content plan based on the blog contents and post type.
        
        Args:
            blog_contents: List of analyzed blog contents
            post_type: Type of post to generate (Twitter or LinkedIn)
            
        Returns:
            Union[Thread, LinkedInPost]: The generated content plan
            
        Raises:
            ValueError: If post_type is unsupported
        """
        try:
            # Combine all blog contents into a single string
            combined_content = "\n\n".join([
                f"Title: {content.title}\n\n{content.content}"
                for content in blog_contents
            ])

            # Use the content writer to generate and improve the content
            response = self.content_writer.run(
                content=combined_content,
                is_twitter=(post_type == PostType.TWITTER)
            )

            if not response.content:
                raise ValueError("Failed to generate content plan")

            return response.content

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

    def run(self, topic: str, post_type: PostType) -> RunResponse:
        """
        Runs the content planning workflow.
        
        Args:
            topic: The topic to research and create content for
            post_type: Type of post to generate (Twitter or LinkedIn)
            
        Returns:
            RunResponse: The workflow response containing the content plan
        """
        try:
            # First, use the research team to find and analyze content
            research_response = self.research_team.run(topic=topic)
            if not research_response.content or not research_response.content.get("blog_contents"):
                return RunResponse(
                    content="No relevant content found for the topic.",
                    event=RunEvent.workflow_completed
                )
            
            blog_contents = research_response.content["blog_contents"]
            
            # Generate content plan based on the analyzed content
            content_plan = self.generate_plan(blog_contents, post_type)
            if not content_plan:
                return RunResponse(
                    content="Failed to generate content plan.",
                    event=RunEvent.workflow_completed
                )
            
            # Schedule and publish the content
            response = self.schedule_and_publish(content_plan, post_type)
            
            # Create a preview of the content using the preview module
            preview_html = create_preview(content_plan)
            
            return RunResponse(
                content=preview_html,
                event=RunEvent.workflow_completed
            )
        except Exception as e:
            logger.error(f"Error in content planning workflow: {str(e)}")
            return RunResponse(
                content=f"Error: {str(e)}",
                event=RunEvent.workflow_completed
            )


if __name__ == "__main__":
    import random
    from rich.prompt import Prompt

    # Example research topics
    example_topics = [
        "The Future of AI in Healthcare",
        "Sustainable Living in 2024",
        "The Impact of Climate Change",
        "Understanding Quantum Computing",
        "The Evolution of Social Media",
        "Mental Health in the Digital Age",
        "The Role of Blockchain",
        "Remote Work Best Practices",
    ]

    # Get topic from user
    topic = Prompt.ask(
        "[bold]Enter a research topic[/bold] (or press Enter for a random example)\n✨",
        default=random.choice(example_topics),
    )

    # Convert the topic to a URL-safe string for use in session_id
    url_safe_topic = topic.lower().replace(" ", "-")

    # Initialize the content planning workflow
    workflow = ContentPlanningWorkflow(
        session_id=f"content-plan-{url_safe_topic}",
        debug_mode=True,
    )

    # Ask user for post type
    post_type = Prompt.ask(
        "[bold]Choose post type[/bold]",
        choices=["twitter", "linkedin"],
        default="linkedin",
    ).upper()

    # Run the workflow
    console = Console()
    with console.status("[bold green]Planning content...") as status:
        response = workflow.run(topic=topic, post_type=PostType[post_type])

    # Create a temporary directory to store the HTML file
    temp_dir = tempfile.mkdtemp()
    html_file = Path(temp_dir) / "response.html"
    
    # Write the HTML content to a file
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(str(response.content))
    
    # Open the HTML file in the default browser
    webbrowser.open(f"file://{html_file}")
    
    logger.info(f"Response saved to: {html_file}")
    logger.info("Content preview opened in browser")
