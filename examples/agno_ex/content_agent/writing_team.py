"""
Content Writing Module

This module handles content writing with self-evaluation for Twitter and LinkedIn content.
"""

from typing import Union
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.utils.log import logger
from agno.workflow import Workflow
from pydantic import BaseModel, Field

from models import Thread, LinkedInPost
from prompts import agents_config, tasks_config


class Feedback(BaseModel):
    """Feedback model for content evaluation."""
    content: Union[Thread, LinkedInPost] = Field(description="The content that needs feedback")
    feedback: str = Field(description="The detailed feedback on the content")
    score: int = Field(description="The score of the content from 0 to 10")


class ContentWriterTeam(Workflow):
    """
    A workflow that creates and self-evaluates social media content.
    Uses a writer-reviewer pattern to iteratively improve content quality.
    """

    description: str = "Content Writer with Self-Evaluation"

    # Twitter Thread Writer: Creates engaging Twitter threads
    twitter_writer: Agent = Agent(
        name="Twitter Thread Writer",
        model=OpenAIChat(id="gpt-4o"),
        description=f"{agents_config['twitter_thread_planner']['role']} - {agents_config['twitter_thread_planner']['goal']}",
        instructions=[
            f"{agents_config['twitter_thread_planner']['backstory']}",
            tasks_config["create_twitter_thread_plan"]["description"],
        ],
        response_model=Thread,
        debug_mode=True,
    )

    # LinkedIn Post Writer: Creates professional LinkedIn posts
    linkedin_writer: Agent = Agent(
        name="LinkedIn Post Writer",
        model=OpenAIChat(id="gpt-4o"),
        description=f"{agents_config['linkedin_post_planner']['role']} - {agents_config['linkedin_post_planner']['goal']}",
        instructions=[
            f"{agents_config['linkedin_post_planner']['backstory']}",
            tasks_config["create_linkedin_post_plan"]["description"],
        ],
        response_model=LinkedInPost,
        debug_mode=True,
    )

    # Content Reviewer: Evaluates and provides feedback on content
    content_reviewer: Agent = Agent(
        name="Content Reviewer",
        model=OpenAIChat(id="gpt-4o"),
        description="Senior Content Strategist specializing in social media optimization",
        instructions=[
            "You are a senior content strategist with expertise in social media content optimization.",
            "Review content for engagement, clarity, and platform-specific best practices.",
            "For Twitter threads:",
            "- Ensure the hook tweet is compelling",
            "- Check thread cohesion and flow",
            "- Verify each tweet can stand alone",
            "- Confirm proper thread length (5-7 tweets ideal)",
            "For LinkedIn posts:",
            "- Verify professional tone and formatting",
            "- Check for clear value proposition",
            "- Ensure appropriate length (1300-1700 characters)",
            "- Validate proper use of line breaks and sections",
            "General criteria:",
            "- Content relevance and accuracy",
            "- Grammar and spelling",
            "- Engagement potential",
            "- Call-to-action effectiveness",
            "Provide specific, actionable feedback for improvement."
        ],
        response_model=Feedback,
        debug_mode=True,
    )

    def write_twitter_thread(self, content: str) -> Thread:
        """
        Creates a Twitter thread with self-evaluation and improvement.
        
        Args:
            content: The content to base the thread on
            
        Returns:
            Thread: The final, improved Twitter thread
        """
        logger.info("Creating Twitter thread...")
        max_tries = 3
        
        # Initial thread creation
        response = self.twitter_writer.run(content)
        thread = response.content if isinstance(response.content, Thread) else None
        
        if not thread:
            raise ValueError("Failed to generate initial Twitter thread")

        # Iterative improvement based on feedback
        for attempt in range(max_tries):
            feedback = self.content_reviewer.run(thread)
            if not isinstance(feedback.content, Feedback):
                continue
                
            if feedback.content.score >= 8:
                logger.info(f"Thread achieved good quality score: {feedback.content.score}")
                break
                
            logger.info(f"Improving thread based on feedback (attempt {attempt + 1})")
            improvement_prompt = (
                f"Please improve this Twitter thread based on the following feedback:\n"
                f"{feedback.content.feedback}\n\n"
                f"Original thread:\n{thread.model_dump_json()}"
            )
            
            response = self.twitter_writer.run(improvement_prompt)
            if isinstance(response.content, Thread):
                thread = response.content

        return thread

    def write_linkedin_post(self, content: str) -> LinkedInPost:
        """
        Creates a LinkedIn post with self-evaluation and improvement.
        
        Args:
            content: The content to base the post on
            
        Returns:
            LinkedInPost: The final, improved LinkedIn post
        """
        logger.info("Creating LinkedIn post...")
        max_tries = 3
        
        # Initial post creation
        response = self.linkedin_writer.run(content)
        post = response.content if isinstance(response.content, LinkedInPost) else None
        
        if not post:
            raise ValueError("Failed to generate initial LinkedIn post")

        # Iterative improvement based on feedback
        for attempt in range(max_tries):
            feedback = self.content_reviewer.run(post)
            if not isinstance(feedback.content, Feedback):
                continue
                
            if feedback.content.score >= 8:
                logger.info(f"Post achieved good quality score: {feedback.content.score}")
                break
                
            logger.info(f"Improving post based on feedback (attempt {attempt + 1})")
            improvement_prompt = (
                f"Please improve this LinkedIn post based on the following feedback:\n"
                f"{feedback.content.feedback}\n\n"
                f"Original post:\n{post.model_dump_json()}"
            )
            
            response = self.linkedin_writer.run(improvement_prompt)
            if isinstance(response.content, LinkedInPost):
                post = response.content

        return post

    def run(self, content: str, is_twitter: bool = False) -> RunResponse:
        """
        Runs the content writing workflow.
        
        Args:
            content: The content to base the social media post on
            is_twitter: Whether to create a Twitter thread (True) or LinkedIn post (False)
            
        Returns:
            RunResponse: The workflow response containing the final content
        """
        try:
            if is_twitter:
                final_content = self.write_twitter_thread(content)
            else:
                final_content = self.write_linkedin_post(content)
                
            return RunResponse(content=final_content)
        except Exception as e:
            logger.error(f"Error in content writing workflow: {str(e)}")
            return RunResponse(content=f"Error: {str(e)}")


if __name__ == "__main__":
    # Test the content writer
    writer = ContentWriterTeam(debug_mode=True)
    
    test_content = """
    The Rise of AI in Healthcare
    
    Artificial Intelligence is revolutionizing healthcare through improved diagnostics,
    personalized treatment plans, and efficient patient care. Machine learning algorithms
    can now detect diseases earlier and more accurately than traditional methods.
    """
    
    # Test Twitter thread
    twitter_response = writer.run(test_content, is_twitter=True)
    print("\nTwitter Thread:")
    print(twitter_response.content)
    
    # Test LinkedIn post
    linkedin_response = writer.run(test_content, is_twitter=False)
    print("\nLinkedIn Post:")
    print(linkedin_response.content) 