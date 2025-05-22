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
from firecrawl import FirecrawlApp
from agno.tools.reasoning import ReasoningTools


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

    def run_research(self, topic: str, use_cache: bool = True) -> RunResponse:
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
    from agno.models.groq import Groq
    from agno.tools.reasoning import ReasoningTools
    from firecrawl import FirecrawlApp
    import os

    # Example research topics
    example_topics = [
        "electric vehicles",
        "study tools"
    ]

    # Get topic from user
    topic = Prompt.ask(
        "[bold]Enter a research topic[/bold] (or press Enter for a random example)\n✨",
        default=random.choice(example_topics),
    )

    console = Console()
    with console.status("[bold green]Researching content...") as status:
        # 1. Use agent with DuckDuckGoTools and ReasoningTools to get main topics and keyword suggestions
        from agno.agent import Agent
        from agno.tools.duckduckgo import DuckDuckGoTools
        from agno.tools.reasoning import ReasoningTools
        agent_search = Agent(
            model=Groq(id="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY")),
            tools=[DuckDuckGoTools(), ReasoningTools()],
            description="You are a research assistant that uses web search and reasoning to extract main topics and suggest search keywords.",
            show_tool_calls=True,
            markdown=True,
        )
        initial_prompt = f"""
        Use DuckDuckGo to search for the topic: '{topic}'.
        Based on the top results, do the following:
        1. Extract and list the 5-8 main topics or themes discussed (numbered list, each with a short description).
        2. For each main topic, suggest 5 effective search keywords or queries (as a sub-list, one per line, no explanations).
        Format:
        1. [Topic name]: [Description]
           - keyword 1
           - keyword 2
           ...
        2. ...
        """
        initial_response = agent_search.run(initial_prompt)
        initial_text = initial_response.content if hasattr(initial_response, 'content') else str(initial_response)
        console.print("\n[bold green]Main Topics and Suggested Keywords:[/bold green]\n")
        console.print(Markdown(initial_text))

        # Parse topics and keywords from the agent's response
        import re
        topic_blocks = re.findall(r"\d+\.\s*(.+?):\s*(.+?)(?=(?:\n\d+\.|\Z))", initial_text, re.DOTALL)
        topics_and_keywords = []
        for block in topic_blocks:
            topic_name = block[0].strip()
            desc_and_keywords = block[1].strip().split('\n')
            description = desc_and_keywords[0].strip()
            keywords = [kw.strip('- ').strip() for kw in desc_and_keywords[1:] if kw.strip()]
            topics_and_keywords.append({"topic": topic_name, "description": description, "keywords": keywords})
        if not topics_and_keywords:
            console.print("[red]Could not parse topics and keywords from the agent's response.[/red]")
            exit(1)

        # Let user select which topics/keywords to investigate further
        topic_names = [tk["topic"] for tk in topics_and_keywords]
        selected = Prompt.ask(
            "[bold]Which topic(s) do you want to investigate further?[/bold] (comma-separated numbers)",
            choices=[str(i+1) for i in range(len(topic_names))],
            default="1"
        )
        selected_indices = [int(i.strip())-1 for i in selected.split(",") if i.strip().isdigit() and 0 <= int(i.strip())-1 < len(topic_names)]
        selected_topics = [topics_and_keywords[i] for i in selected_indices]
        console.print(f"\n[bold blue]You selected to investigate:[/bold blue] {', '.join(t['topic'] for t in selected_topics)}\n")

        # Let user select which keywords to use for each selected topic
        selected_keywords = []
        for t in selected_topics:
            if not t["keywords"]:
                continue
            selected_keywords.extend(t["keywords"])

        if not selected_keywords:
            console.print("[red]No keywords found for further investigation.[/red]")
            exit(1)

        # 2. For each selected keyword, redo the search, crawl, and analyze (as before)
        for keyword in selected_keywords:
            console.print(f"\n[bold yellow]Researching keyword:[/bold yellow] {keyword}")
            ddg_results = DuckDuckGoTools().duckduckgo_search(query=keyword, max_results=5)
            if isinstance(ddg_results, str):
                try:
                    ddg_results = json.loads(ddg_results)
                except Exception:
                    ddg_results = []
            if not isinstance(ddg_results, list):
                ddg_results = []
            crawled_contents = []
            for result in ddg_results:
                url = result.get("href") or result.get("link")
                title = result.get("title", "No Title")
                if not url:
                    continue
                try:
                    crawl_result = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY")).scrape_url(url, params={"formats": ["markdown"]})
                    content = crawl_result.get("markdown") if crawl_result else None
                    if content:
                        crawled_contents.append({
                            "title": title,
                            "url": url,
                            "content": content
                        })
                except Exception as e:
                    console.print(f"[yellow]Failed to crawl {url}: {e}[/yellow]")
            if not crawled_contents:
                console.print(f"[red]No content could be crawled for keyword '{keyword}'.[/red]")
                continue
            combined_content = "\n\n".join(f"Title: {c['title']}\nURL: {c['url']}\nContent:\n{c['content']}" for c in crawled_contents)
            organize_prompt = f"""
            Organize and summarize the following research results for the keyword '{keyword}'.
            - Extract the main findings, trends, and insights.
            - List any important facts, statistics, or unique perspectives.
            - Provide a detailed summary of the content, do not skip any important details.
            - Output in markdown with clear sections: Findings, Facts, Perspectives, Summary.
            Content:
            {combined_content}
            """
            organize_response = agent_search.run(organize_prompt)
            organize_text = organize_response.content if hasattr(organize_response, 'content') else str(organize_response)
            console.print(f"\n[bold green]Organized Research Results for '{keyword}':[/bold green]\n")
            console.print(Markdown(organize_text)) 