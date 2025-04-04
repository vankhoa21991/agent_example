"""
Data Models

This module defines the data models used throughout the content planning workflow.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


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