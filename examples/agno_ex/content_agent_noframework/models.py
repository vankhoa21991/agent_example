from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class NewsArticle:
    """A news article with title, content, and source"""
    title: str
    content: str
    url: str
    source: str = ""
    source_type: str = "web"  # "web" or "wikipedia"
    
@dataclass
class BlogContent:
    """Content for a blog post"""
    title: str
    content: str
    platform: str
    topic: str
    sources: List[NewsArticle] = field(default_factory=list)
    
@dataclass
class RunResponse:
    """Response from a run operation"""
    status: str = "success"
    message: str = ""
    content: Optional[Dict[str, Any]] = None

def create_error_response(message: str) -> RunResponse:
    """Create an error response with the given message"""
    return RunResponse(
        status="error",
        message=message,
        content=None
    )

def create_success_response(content: Dict[str, Any], message: str = "") -> RunResponse:
    """Create a success response with the given content and message"""
    return RunResponse(
        status="success",
        message=message,
        content=content
    ) 