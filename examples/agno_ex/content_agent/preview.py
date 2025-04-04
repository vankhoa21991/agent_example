"""
Content Preview Generator

This module provides functionality to create beautiful HTML previews of social media content plans.
"""

from typing import Union
from agno.utils.log import logger

from models import Thread, LinkedInPost


def create_preview(content_plan: Union[Thread, LinkedInPost]) -> str:
    """
    Creates a beautiful HTML preview of the content plan with markdown support.
    
    Args:
        content_plan: The content plan to preview (Thread or LinkedInPost)
        
    Returns:
        str: HTML content for preview
    """
    try:
        if isinstance(content_plan, Thread):
            # Create HTML for Twitter thread
            tweets_html = []
            for tweet in content_plan.tweets:
                media_html = ""
                if tweet.media_urls:
                    media_html = '<div class="tweet-media">' + ''.join([
                        f'<img src="{url}" alt="Media">' for url in tweet.media_urls
                    ]) + '</div>'
                
                tweet_html = f'''
                <div class="tweet {'hook' if tweet.is_hook else ''}">
                    <div class="tweet-content">
                        <div class="tweet-header">
                            <img src="https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png" class="profile-pic">
                            <div class="tweet-info">
                                <span class="username">@ContentCreator</span>
                                <span class="handle">@content_creator</span>
                            </div>
                        </div>
                        <div class="tweet-text markdown-body">
                            {tweet.content}
                        </div>
                        {media_html}
                    </div>
                </div>
                '''
                tweets_html.append(tweet_html)
            
            html_content = f'''
            <div class="thread-preview">
                <h2>Twitter Thread Preview</h2>
                <h3>Topic: {content_plan.topic}</h3>
                <div class="tweets">
                    {''.join(tweets_html)}
                </div>
            </div>
            '''
        elif isinstance(content_plan, LinkedInPost):
            # Create HTML for LinkedIn post
            media_html = ""
            if content_plan.media_url:
                media_html = '<div class="post-media">' + ''.join([
                    f'<img src="{url}" alt="Media">' for url in content_plan.media_url
                ]) + '</div>'
            
            html_content = f'''
            <div class="linkedin-preview">
                <div class="post-content">
                    <div class="post-header">
                        <img src="https://static.licdn.com/sc/h/1c5u578iilxfi4m4dvc4q810q" class="profile-pic">
                        <div class="post-info">
                            <span class="name">Content Creator</span>
                            <span class="title">Professional Content Writer</span>
                        </div>
                    </div>
                    <div class="post-text markdown-body">
                        {content_plan.content}
                    </div>
                    {media_html}
                    <div class="post-footer">
                        <div class="engagement">
                            <span class="likes">👍 0</span>
                            <span class="comments">💬 0</span>
                            <span class="shares">↗️ 0</span>
                        </div>
                    </div>
                </div>
            </div>
            '''
        else:
            raise ValueError(f"Unsupported content plan type: {type(content_plan)}")

        # Add GitHub markdown CSS and custom styling
        full_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Content Preview</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
            <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    background-color: #f5f5f5;
                    margin: 0;
                    padding: 20px;
                }}
                .markdown-body {{
                    box-sizing: border-box;
                    min-width: 200px;
                    max-width: 980px;
                    margin: 0 auto;
                    padding: 20px;
                    background: transparent;
                }}
                .thread-preview, .linkedin-preview {{
                    max-width: 600px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .tweet, .post-content {{
                    padding: 16px;
                    border-bottom: 1px solid #e1e8ed;
                }}
                .tweet:last-child, .post-content:last-child {{
                    border-bottom: none;
                }}
                .tweet-header, .post-header {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 12px;
                }}
                .profile-pic {{
                    width: 48px;
                    height: 48px;
                    border-radius: 50%;
                    margin-right: 12px;
                }}
                .tweet-info, .post-info {{
                    display: flex;
                    flex-direction: column;
                }}
                .username, .name {{
                    font-weight: bold;
                    color: #14171a;
                }}
                .handle, .title {{
                    color: #657786;
                    font-size: 14px;
                }}
                .tweet-text, .post-text {{
                    font-size: 15px;
                    line-height: 1.5;
                    color: #14171a;
                    margin-bottom: 12px;
                    white-space: pre-wrap;
                }}
                .tweet-media, .post-media {{
                    margin-top: 12px;
                    border-radius: 12px;
                    overflow: hidden;
                }}
                .tweet-media img, .post-media img {{
                    width: 100%;
                    border-radius: 12px;
                }}
                .hook {{
                    background-color: #f7f9fa;
                }}
                h2 {{
                    padding: 16px;
                    margin: 0;
                    background: #f7f9fa;
                    border-bottom: 1px solid #e1e8ed;
                    font-size: 18px;
                    color: #14171a;
                }}
                h3 {{
                    padding: 0 16px;
                    margin: 12px 0;
                    color: #657786;
                    font-size: 16px;
                }}
                .error {{
                    color: #e0245e;
                    padding: 16px;
                    text-align: center;
                }}
                .markdown-body strong {{
                    font-weight: bold;
                    color: #14171a;
                }}
                .markdown-body em {{
                    font-style: italic;
                }}
                .post-footer {{
                    margin-top: 16px;
                    padding-top: 12px;
                    border-top: 1px solid #e1e8ed;
                }}
                .engagement {{
                    display: flex;
                    gap: 16px;
                    color: #657786;
                    font-size: 14px;
                }}
                .engagement span {{
                    display: flex;
                    align-items: center;
                    gap: 4px;
                }}
                .linkedin-preview {{
                    background: #fff;
                    border: 1px solid #e1e8ed;
                }}
                .post-text p {{
                    margin: 0 0 12px 0;
                }}
                .post-text p:last-child {{
                    margin-bottom: 0;
                }}
            </style>
        </head>
        <body>
            <div class="markdown-body">
                {html_content}
            </div>
            <script>
                // Configure marked
                marked.setOptions({{
                    breaks: true,
                    gfm: true,
                    headerIds: false,
                    mangle: false
                }});

                // Convert all markdown content
                document.querySelectorAll('.tweet-text, .post-text').forEach(element => {{
                    element.innerHTML = marked.parse(element.textContent.trim());
                }});
            </script>
        </body>
        </html>
        '''
        return full_html
    except Exception as e:
        logger.error(f"Error creating preview: {str(e)}")
        return f"<div class='error'>Error creating preview: {str(e)}</div>" 