import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq
from firecrawl import FirecrawlApp

from models import NewsArticle, create_error_response, create_success_response, RunResponse
from agno.tools.duckduckgo import DuckDuckGoTools

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create tool instances globally
ddg_search = DuckDuckGoTools()

class ResearchTeam:
    """Handles research tasks using Groq directly instead of agents"""
    
    def __init__(self):
        # Use Groq for LLM functionality
        self.model = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.search_cache = {}  # Cache search results
        
        # Initialize Firecrawl
        self.firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        
    def crawl_url(self, url: str) -> Dict[str, Any]:
        """Crawl a specific URL using Firecrawl to get detailed content"""
        try:
            logger.info(f"Crawling URL: {url}")
            
            # Use Firecrawl to get detailed content
            result = self.firecrawl.scrape_url(url, params={'formats': ['markdown']})
            
            if result and 'markdown' in result:
                logger.info(f"Successfully crawled URL: {url}")
                
                # Create summary of crawled content using Groq
                summary = self.summarize_crawled_content(result['markdown'], url)
                
                return {
                    "url": url,
                    "content": summary,
                    "success": True,
                    "raw_content": result['markdown'][:5000]  # Limit raw content to 5000 chars
                }
            else:
                logger.warning(f"No markdown content found for URL: {url}")
                return {"url": url, "content": "", "success": False}
                
        except Exception as e:
            logger.error(f"Error crawling URL {url}: {str(e)}")
            return {"url": url, "content": "", "success": False, "error": str(e)}
    
    def summarize_crawled_content(self, content: str, url: str) -> str:
        """Summarize crawled content to extract the most relevant information"""
        try:
            # Limit content length to avoid token limits
            if len(content) > 10000:
                content = content[:10000] + "..."
                
            prompt = f"""
            Please provide a concise summary of the following website content from {url}.
            Focus on extracting the key information that would be most relevant for content research.
            Your summary should:
            - Be around 2-3 paragraphs
            - Highlight the main topics and key points
            - Include important facts, statistics, or data if present
            - Maintain objectivity
            
            Content:
            {content}
            """
            
            completion = self.model.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a content research assistant that specializes in extracting and summarizing key information from websites."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            summary = completion.choices[0].message.content
            logger.info(f"Generated summary for {url}")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing content: {str(e)}")
            return "Error generating summary"
    
    def search_content(self, topic: str) -> List[Dict[str, Any]]:
        """Search for content related to a topic using DuckDuckGo and Firecrawl"""
        
        # Check cache first
        if topic in self.search_cache:
            logger.info(f"Using cached search results for topic: {topic}")
            return self.search_cache[topic]
            
        logger.info(f"Searching for content about: {topic}")
        
        # Store results
        articles = []
        
        # DuckDuckGo Search - using the same approach as in the example
        try:
            # Simple direct call matching the example
            results = ddg_search.duckduckgo_search(query=topic, max_results=5)  # Reduced to 5 to allow time for crawling
            logger.info(f"Raw DuckDuckGo results type: {type(results)}")
            
            # Process search results to extract URLs
            urls_to_crawl = []
            
            # If results is a JSON string, parse it into a Python object
            if isinstance(results, str):
                try:
                    # Parse JSON string to Python object
                    parsed_results = json.loads(results)
                    logger.info(f"Parsed results into type: {type(parsed_results)}")
                    
                    # If parsed result is a list of dictionaries, process each item
                    if isinstance(parsed_results, list):
                        for item in parsed_results:
                            if isinstance(item, dict):
                                url = item.get("href", "")
                                if url:
                                    urls_to_crawl.append(url)
                                
                                articles.append({
                                    "title": item.get("title", "No Title"),
                                    "content": item.get("body", ""),
                                    "url": url,
                                    "source": "DuckDuckGo",
                                    "source_type": "web"
                                })
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON result: {e}")
                    # Add the string as a single result if JSON parsing fails
                    articles.append({
                        "title": "DuckDuckGo Result",
                        "content": results[:500],  # Limit to first 500 chars
                        "url": "",
                        "source": "DuckDuckGo",
                        "source_type": "web"
                    })
            # If results is already a list, process it directly
            elif isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        url = item.get("href", item.get("link", ""))
                        if url:
                            urls_to_crawl.append(url)
                            
                        articles.append({
                            "title": item.get("title", "No Title"),
                            "content": item.get("body", item.get("snippet", "")),
                            "url": url,
                            "source": "DuckDuckGo",
                            "source_type": "web"
                        })
                    elif isinstance(item, str):
                        articles.append({
                            "title": "Result",
                            "content": item,
                            "url": "",
                            "source": "DuckDuckGo",
                            "source_type": "web"
                        })
            # If results is a dictionary, add it as a single result
            elif isinstance(results, dict):
                url = results.get("href", results.get("link", ""))
                if url:
                    urls_to_crawl.append(url)
                    
                articles.append({
                    "title": results.get("title", "DuckDuckGo Result"),
                    "content": results.get("body", results.get("snippet", "")),
                    "url": url,
                    "source": "DuckDuckGo",
                    "source_type": "web"
                })
            
            logger.info(f"Found {len(articles)} results from DuckDuckGo")
            logger.info(f"Found {len(urls_to_crawl)} URLs to crawl")
            
            # Crawl URLs to get detailed content
            if urls_to_crawl and os.getenv("FIRECRAWL_API_KEY"):
                logger.info("Starting to crawl URLs with Firecrawl")
                
                for url in urls_to_crawl[:3]:  # Limit to first 3 URLs to avoid rate limits
                    try:
                        crawl_result = self.crawl_url(url)
                        
                        if crawl_result["success"]:
                            # Add as a separate article with enhanced content
                            articles.append({
                                "title": f"Detailed: {url}",
                                "content": crawl_result["content"],
                                "url": url,
                                "source": "Firecrawl",
                                "source_type": "web",
                                "raw_content": crawl_result.get("raw_content", "")
                            })
                            logger.info(f"Added detailed content for URL: {url}")
                    except Exception as crawl_error:
                        logger.error(f"Error during crawl of {url}: {str(crawl_error)}")
                
                logger.info("Completed crawling URLs")
            else:
                logger.warning("Skipping URL crawling: No URLs found or FIRECRAWL_API_KEY not set")
            
            # If no results, add a default placeholder
            if not articles:
                logger.warning("No results found, adding placeholder")
                articles.append({
                    "title": "No Results",
                    "content": f"No information found for '{topic}'. Please try a different search term.",
                    "url": "",
                    "source": "DuckDuckGo",
                    "source_type": "web"
                })
                
        except Exception as e:
            logger.error(f"Error in research process: {str(e)}")
            
            # If error occurs, add a placeholder result
            articles.append({
                "title": "Search Error", 
                "content": f"There was an error performing the search: {str(e)}. Please try again later.",
                "url": "", 
                "source": "Error",
                "source_type": "web"
            })
        
        # Cache the results
        self.search_cache[topic] = articles
        
        return articles
    
    def analyze_content(self, topic: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content using Groq model"""
        
        # Format articles for the prompt
        formatted_articles = []
        for i, article in enumerate(articles):
            source_icon = "🔍" if article.get('source') == "Firecrawl" else "🌐"
            formatted_articles.append(
                f"Article {i+1} {source_icon} - {article['title']}\n"
                f"Source: {article['source']}\n"
                f"URL: {article['url']}\n"
                f"Content: {article['content']}\n"
            )
        
        articles_text = "\n".join(formatted_articles)
        
        # Create prompt for analysis
        prompt = f"""
        You are a content researcher and analyst. I need you to analyze the following articles 
        about "{topic}" and provide a summary of key points and insights.
        
        ARTICLES:
        {articles_text}
        
        Please provide:
        1. A list of 3-5 key insights from these articles
        2. A list of important facts or statistics
        3. A list of unique perspectives or angles to cover
        4. A summary of the overall narrative or story
        
        Format your response as a JSON object with the following keys:
        - insights: array of strings
        - facts: array of strings
        - perspectives: array of strings
        - summary: string
        
        Respond ONLY with valid JSON.
        """
        
        try:
            completion = self.model.chat.completions.create(
                model="llama3-70b-8192",  # Using Llama3-70B model
                messages=[
                    {"role": "system", "content": "You are a helpful content analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            # Extract the JSON response
            response_text = completion.choices[0].message.content
            
            # Parse JSON and handle potential errors
            try:
                analysis = json.loads(response_text)
                return analysis
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from model response")
                # Try to extract JSON with regex if standard parsing fails
                import re
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                # Return a default structure with the raw response
                return {
                    "insights": ["Error: Could not parse analysis"],
                    "facts": [],
                    "perspectives": [],
                    "summary": "Error analyzing content: " + response_text[:100]
                }
                
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            return {
                "insights": ["Error: " + str(e)],
                "facts": [],
                "perspectives": [],
                "summary": "Error analyzing content"
            }
    
    def run(self, topic: str) -> RunResponse:
        """Main method to run research and analysis on a topic"""
        try:
            # Step 1: Search for content
            articles = self.search_content(topic)
            
            if not articles:
                return create_error_response("No articles found for the given topic")
            
            # Step 2: Analyze the content
            analysis = self.analyze_content(topic, articles)
            
            # Step 3: Format the results
            news_articles = [
                NewsArticle(
                    title=article["title"],
                    content=article["content"],
                    url=article["url"],
                    source=article["source"],
                    source_type=article.get("source_type", "web")
                )
                for article in articles
            ]
            
            response_content = {
                "articles": [article.__dict__ for article in news_articles],
                "analysis": analysis
            }
            
            return create_success_response(response_content)
            
        except Exception as e:
            logger.error(f"Research error: {str(e)}")
            return create_error_response(f"Research error: {str(e)}")

# Simple test function that matches the example
def test_research():
    try:
        # Simple test similar to the example
        print("Testing direct DuckDuckGo search:")
        query = 'Artificial Intelligence trends'
        direct_result = ddg_search.duckduckgo_search(query=query, max_results=5)
        print(f"Direct search result type: {type(direct_result)}")
        print(f"Direct search result count: {len(direct_result) if isinstance(direct_result, list) else 1}")
        
        # Test with research team
        print("\nTesting with ResearchTeam:")
        research_team = ResearchTeam()
        response = research_team.run("artificial intelligence trends")
        print(json.dumps(response.__dict__, indent=2))
    except Exception as e:
        print(f"Test error: {str(e)}")

if __name__ == "__main__":
    test_research() 