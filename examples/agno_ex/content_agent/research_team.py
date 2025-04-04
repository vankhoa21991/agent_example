"""🔍 Research Team - Advanced Content Research and Analysis

This module implements a research team that combines web search and content analysis capabilities
to find and analyze high-quality content for blog posts and social media content.

The team consists of:
1. ContentSearcher-X: Finds authoritative and relevant content
2. ContentAnalyzer-X: Extracts and analyzes blog content
"""

import json
from typing import Dict, List, Optional, Union
from textwrap import dedent
import os
from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from agno.run.response import RunEvent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.firecrawl import FirecrawlTools
from agno.utils.log import logger
from agno.workflow import Workflow
from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    """Represents a news article found during research."""
    title: str = Field(..., description="Title of the article")
    url: str = Field(..., description="URL of the article")
    summary: Optional[str] = Field(None, description="Summary of the article content")


class SearchResults(BaseModel):
    """Collection of articles found during research."""
    articles: List[NewsArticle]


class BlogContent(BaseModel):
    """Represents analyzed blog content."""
    title: str = Field(..., description="Title of the blog post")
    url: str = Field(..., description="URL of the blog post")
    content: str = Field(..., description="Blog content in markdown format")
    key_points: List[str] = Field(..., description="Key points extracted from the content")
    summary: str = Field(..., description="Summary of the blog content")


class ResearchTeam(Workflow):
    """A team of agents working together to research and analyze content."""

    description: str = dedent("""\
    An intelligent research team that combines web search and content analysis capabilities
    to find and analyze high-quality content. The team works collaboratively to:
    1. Find authoritative and relevant content
    2. Extract and analyze blog content
    3. Generate insights and summaries
    """)

    # Content Searcher Agent
    searcher: Agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY")),
        tools=[DuckDuckGoTools()],
        description=dedent("""\
        You are ContentSearcher-X, an expert in finding high-quality content and sources.
        Your expertise includes:

        - Finding authoritative sources
        - Evaluating content relevance
        - Identifying trending topics
        - Discovering unique perspectives
        - Ensuring comprehensive coverage\
        """),
        instructions=dedent("""\
        1. Search Strategy 🔍
           - Find 5-7 relevant sources
           - Prioritize recent content
           - Look for authoritative sources
           - Consider different perspectives

        2. Source Evaluation 📊
           - Verify source credibility
           - Check publication dates
           - Assess content quality
           - Validate information

        3. Content Selection 🎯
           - Choose the most relevant source
           - Ensure comprehensive coverage
           - Look for unique insights
           - Consider engagement potential\
        """),
        response_model=None,  # Remove response model to allow tool usage
        show_tool_calls=True,
        markdown=True,
    )

    # Content Analyzer Agent
    analyzer: Agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY")),
        tools=[FirecrawlTools(scrape=True, crawl=False)],
        description=dedent("""\
        You are ContentAnalyzer-X, an expert in extracting and analyzing blog content.
        Your expertise includes:

        - Efficient content extraction and parsing
        - Key information identification
        - Content structure analysis
        - Main points extraction
        - Source attribution preservation\
        """),
        instructions=dedent("""\
        1. Content Extraction 📑
           - Extract the main content and title
           - Preserve important sections and structure
           - Maintain formatting and links
           - Handle different content types

        2. Content Analysis 🔍
           - Identify key points and takeaways
           - Extract relevant quotes and statistics
           - Note important context and background
           - Mark sections for potential adaptation

        3. Quality Control ✅
           - Verify content completeness
           - Ensure proper formatting
           - Check for missing elements
           - Validate source attribution\
        """),
        response_model=None,  # Remove response model to allow tool usage
        show_tool_calls=True,
        markdown=True,
    )

    def search_content(self, topic: str, use_cache: bool = True) -> List[NewsArticle]:
        """
        Searches for relevant content based on the given topic.
        
        Args:
            topic: The topic to search for
            use_cache: Whether to use cached content if available
            
        Returns:
            List[NewsArticle]: List of relevant articles found
            
        Raises:
            ValueError: If the searcher returns unexpected content type
            Exception: If there's an error during searching
        """
        try:
            if use_cache and topic in self.session_state.get("search_results", {}):
                logger.info(f"Using cached search results for topic: {topic}")
                return self.session_state["search_results"][topic]
            
            response: RunResponse = self.searcher.run(topic)
            if not response or not response.content:
                raise ValueError("Empty response received from searcher")
                
            # Handle string response from Groq
            if isinstance(response.content, str):
                try:
                    # Try to parse the response as JSON
                    data = json.loads(response.content)
                    if "articles" in data:
                        articles = [NewsArticle(**article) for article in data["articles"]]
                        self.session_state.setdefault("search_results", {})[topic] = articles
                        return articles
                except json.JSONDecodeError:
                    # If not JSON, create a simple article from the content
                    article = NewsArticle(
                        title=topic,
                        url=response.content,
                        summary="Content found from search"
                    )
                    self.session_state.setdefault("search_results", {})[topic] = [article]
                    return [article]
            
            # Handle dictionary response
            elif isinstance(response.content, dict):
                if "articles" in response.content:
                    articles = [NewsArticle(**article) for article in response.content["articles"]]
                    self.session_state.setdefault("search_results", {})[topic] = articles
                    return articles
            
            raise ValueError(f"Unexpected content type received from searcher: {type(response.content)}")
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            raise

    def analyze_content(self, article: NewsArticle, use_cache: bool = True) -> BlogContent:
        """
        Analyzes the content of a given article.
        
        Args:
            article: The article to analyze
            use_cache: Whether to use cached content if available
            
        Returns:
            BlogContent: The analyzed blog content
            
        Raises:
            ValueError: If the analyzer returns unexpected content type
            Exception: If there's an error during analysis
        """
        try:
            if use_cache and article.url in self.session_state:
                logger.info(f"Using cached analysis for article: {article.url}")
                return self.session_state[article.url]
            
            response: RunResponse = self.analyzer.run(article.url)
            if not response or not response.content:
                raise ValueError("Empty response received from analyzer")
                
            # Handle string response from Groq
            if isinstance(response.content, str):
                try:
                    # Try to parse the response as JSON
                    data = json.loads(response.content)
                    if "title" in data and "content" in data:
                        blog_content = BlogContent(
                            title=data["title"],
                            url=article.url,
                            content=data["content"],
                            key_points=data.get("key_points", []),
                            summary=data.get("summary", "")
                        )
                        self.session_state[article.url] = blog_content
                        return blog_content
                except json.JSONDecodeError:
                    # If not JSON, create a simple blog content from the response
                    blog_content = BlogContent(
                        title=article.title,
                        url=article.url,
                        content=response.content,
                        key_points=["Content extracted from article"],
                        summary=article.summary or "Content analyzed from article"
                    )
                    self.session_state[article.url] = blog_content
                    return blog_content
            
            # Handle dictionary response
            elif isinstance(response.content, dict):
                if "title" in response.content and "content" in response.content:
                    blog_content = BlogContent(
                        title=response.content["title"],
                        url=article.url,
                        content=response.content["content"],
                        key_points=response.content.get("key_points", []),
                        summary=response.content.get("summary", "")
                    )
                    self.session_state[article.url] = blog_content
                    return blog_content
            
            raise ValueError(f"Unexpected content type received from analyzer: {type(response.content)}")
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            raise

    def run(self, topic: str, use_cache: bool = True) -> RunResponse:
        """
        Runs the research team workflow.
        
        Args:
            topic: The topic to research
            use_cache: Whether to use cached content if available
            
        Returns:
            RunResponse: The workflow response containing the list of blog contents
            
        Raises:
            Exception: If there's an error during the workflow
        """
        try:
            # Search for relevant content
            articles = self.search_content(topic, use_cache)
            if not articles:
                logger.warning(f"No articles found for topic: {topic}")
                return RunResponse(
                    content={"blog_contents": []},
                    event=RunEvent.workflow_completed
                )
            
            logger.info(f"Found {len(articles)} articles for topic: {topic}")
            # Analyze each article
            blog_contents = []
            for article in articles:
                try:
                    blog_content = self.analyze_content(article, use_cache)
                    if blog_content:
                        blog_contents.append(blog_content)
                except Exception as e:
                    logger.warning(f"Failed to analyze article {article.url}: {str(e)}")
                    continue
            
            logger.info(f"Analyzed {len(blog_contents)} blog contents for topic: {topic}")
            logger.info(f"Blog contents: {blog_contents}")
            if not blog_contents:
                logger.warning(f"No blog contents were successfully analyzed for topic: {topic}")
                return RunResponse(
                    content={"blog_contents": []},
                    event=RunEvent.workflow_completed
                )
            
            return RunResponse(
                content={"blog_contents": blog_contents},
                event=RunEvent.workflow_completed
            )
        except Exception as e:
            logger.error(f"Error in research team workflow: {str(e)}")
            return RunResponse(
                content={"blog_contents": []},
                event=RunEvent.workflow_completed
            )


if __name__ == "__main__":
    import random
    from rich.prompt import Prompt
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

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

    # Initialize the research team
    team = ResearchTeam(
        session_id=f"research-{topic.lower().replace(' ', '-')}",
        debug_mode=True,
    )

    # Run the research
    console = Console()
    with console.status("[bold green]Researching content...") as status:
        response = team.run(topic)

    # Display results
    console.print("\n[bold green]Research Results:[/bold green]\n")
    
    for content in response.content["blog_contents"]:
        console.print(Panel(
            f"[bold blue]{content.title}[/bold blue]\n\n"
            f"[italic]URL:[/italic] {content.url}\n\n"
            f"[bold]Key Points:[/bold]\n" + "\n".join(f"- {point}" for point in content.key_points) + "\n\n"
            f"[bold]Summary:[/bold]\n{content.summary}\n\n"
            f"[bold]Content Preview:[/bold]\n{content.content[:500]}...",
            title="📄 Blog Content Analysis",
            border_style="blue"
        ))
        console.print("\n") 