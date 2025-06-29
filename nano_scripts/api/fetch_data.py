"""
Nano-script: API Data Fetcher
Atomic script for fetching data from APIs
"""

import requests
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def fetch_api_data(
    url: str,
    method: str = "GET",
    headers: Dict[str, str] = None,
    params: Dict[str, Any] = None,
    data: Dict[str, Any] = None,
    output_file: str = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Fetch data from an API endpoint

    Args:
        url: API endpoint URL
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: HTTP headers
        params: URL parameters
        data: Request body data
        output_file: File to save response to
        timeout: Request timeout in seconds
    """
    try:
        print(f"Fetching data from {url} using {method}")

        # Prepare request
        kwargs = {"timeout": timeout, "headers": headers or {}, "params": params or {}}

        if method.upper() in ["POST", "PUT", "PATCH"] and data:
            kwargs["json"] = data

        # Make request
        response = requests.request(method.upper(), url, **kwargs)
        response.raise_for_status()

        # Parse response
        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"text": response.text, "status_code": response.status_code}

        print(f"Successfully fetched data (status: {response.status_code})")

        # Save to file if requested
        if output_file:
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Response saved to {output_file}")

        return result

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error fetching API data: {e}")
        sys.exit(1)


def fetch_hackernews_top(limit: int = 10) -> Dict[str, Any]:
    """Fetch top stories from Hacker News API"""
    try:
        # Get top story IDs
        top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        story_ids = fetch_api_data(top_stories_url)[:limit]

        stories = []
        for story_id in story_ids:
            story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
            story = fetch_api_data(story_url)
            if story.get("title"):
                stories.append(
                    {
                        "title": story.get("title"),
                        "url": story.get("url"),
                        "score": story.get("score", 0),
                        "by": story.get("by"),
                        "time": story.get("time"),
                    }
                )

        return {"stories": stories, "count": len(stories)}

    except Exception as e:
        print(f"Error fetching Hacker News data: {e}")
        return {"stories": [], "count": 0}


def main():
    parser = argparse.ArgumentParser(description="Fetch data from APIs")
    parser.add_argument("url", nargs="?", help="API endpoint URL")
    parser.add_argument("-m", "--method", default="GET", help="HTTP method")
    parser.add_argument(
        "-H", "--header", action="append", help="HTTP header (key:value)"
    )
    parser.add_argument(
        "-p", "--param", action="append", help="URL parameter (key=value)"
    )
    parser.add_argument("-d", "--data", help="Request body (JSON string)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout")
    parser.add_argument(
        "--hackernews",
        type=int,
        metavar="LIMIT",
        help="Fetch top Hacker News stories (specify limit)",
    )

    args = parser.parse_args()

    # Handle Hacker News shortcut
    if args.hackernews:
        result = fetch_hackernews_top(args.hackernews)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))
        return

    if not args.url:
        parser.error("URL is required unless using --hackernews")

    # Parse headers
    headers = {}
    if args.header:
        for header in args.header:
            key, value = header.split(":", 1)
            headers[key.strip()] = value.strip()

    # Parse parameters
    params = {}
    if args.param:
        for param in args.param:
            key, value = param.split("=", 1)
            params[key.strip()] = value.strip()

    # Parse data
    data = None
    if args.data:
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError:
            print("Error: Invalid JSON in --data parameter")
            sys.exit(1)

    # Fetch data
    result = fetch_api_data(
        url=args.url,
        method=args.method,
        headers=headers,
        params=params,
        data=data,
        output_file=args.output,
        timeout=args.timeout,
    )

    # Print result if not saving to file
    if not args.output:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
