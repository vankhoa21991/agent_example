"""
Example of web crawling using Playwright.
Shows proper error handling and resource cleanup for both synchronous and asynchronous approaches.
"""

import asyncio
import logging
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example URL - using a more reliable public site
URL = 'https://medium.com/heurislabs/building-a-universal-assistant-to-connect-with-any-api-89d7c353e524'

# Async example
async def crawl_async():
    """Asynchronous web crawling example with proper error handling."""
    browser = None
    try:
        async with async_playwright() as pw:
            # Launch browser with options
            browser = await pw.chromium.launch(
                headless=False,  # Headless mode for production
            )
            
            # Create a new page
            page = await browser.new_page()
            
            # Navigate and wait for content
            logger.info(f"Navigating to {URL}")
            await page.goto(URL)
            await page.wait_for_selector('h1')
            
            # Get page content
            content = await page.content()
            title = await page.title()
            
            logger.info(f"Successfully crawled page: {title}")
            print(content)
            # Close browser explicitly
            await browser.close()
            return content
            
    except Exception as e:
        logger.error(f"Error during async crawling: {str(e)}")
        # Ensure browser is closed even if an error occurs
        if browser:
            await browser.close()
        raise
    
# Sync example
def crawl_sync():
    """Synchronous web crawling example with proper error handling."""
    browser = None
    try:
        with sync_playwright() as pw:
            # Launch browser with options
            browser = pw.chromium.launch(
                headless=True,  # Headless mode for production
            )
            
            # Create a new page
            page = browser.new_page()
            
            # Navigate and wait for content
            logger.info(f"Navigating to {URL}")
            page.goto(URL)
            page.wait_for_selector('h1')
            
            # Get page content
            content = page.content()
            title = page.title()
            
            logger.info(f"Successfully crawled page: {title}")
            
            # Close browser explicitly
            browser.close()
            return content
            
    except Exception as e:
        logger.error(f"Error during sync crawling: {str(e)}")
        # Ensure browser is closed even if an error occurs
        if browser:
            browser.close()
        raise

async def main():
    """Main function to demonstrate both async and sync approaches."""
    try:
        # Run async example
        logger.info("Running async example...")
        content_async = await crawl_async()
        print("\nAsync content length:", len(content_async))
        
        # Run sync example
        logger.info("\nRunning sync example...")
        content_sync = crawl_sync()
        print("Sync content length:", len(content_sync))
        
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
