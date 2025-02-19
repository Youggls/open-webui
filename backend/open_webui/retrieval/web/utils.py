import socket
import aiohttp
import asyncio
import urllib.parse
import validators

from concurrent.futures import ThreadPoolExecutor
from typing import Union, Sequence, Iterator, Any, AsyncIterator, Dict, Iterator, List, Dict, Optional

from langchain_community.document_loaders import (
    WebBaseLoader,
)
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_community.document_loaders.firecrawl import FireCrawlLoader

from open_webui.constants import ERROR_MESSAGES
from open_webui.config import ENABLE_RAG_LOCAL_WEB_FETCH
from open_webui.env import SRC_LOG_LEVELS
from typing import Literal
import logging

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


def validate_url(url: Union[str, Sequence[str]]):
    if isinstance(url, str):
        if isinstance(validators.url(url), validators.ValidationError):
            raise ValueError(ERROR_MESSAGES.INVALID_URL)
        if not ENABLE_RAG_LOCAL_WEB_FETCH:
            # Local web fetch is disabled, filter out any URLs that resolve to private IP addresses
            parsed_url = urllib.parse.urlparse(url)
            # Get IPv4 and IPv6 addresses
            ipv4_addresses, ipv6_addresses = resolve_hostname(parsed_url.hostname)
            # Check if any of the resolved addresses are private
            # This is technically still vulnerable to DNS rebinding attacks, as we don't control WebBaseLoader
            for ip in ipv4_addresses:
                if validators.ipv4(ip, private=True):
                    raise ValueError(ERROR_MESSAGES.INVALID_URL)
            for ip in ipv6_addresses:
                if validators.ipv6(ip, private=True):
                    raise ValueError(ERROR_MESSAGES.INVALID_URL)
        return True
    elif isinstance(url, Sequence):
        return all(validate_url(u) for u in url)
    else:
        return False


def safe_validate_urls(url: Sequence[str]) -> Sequence[str]:
    valid_urls = []
    for u in url:
        try:
            if validate_url(u):
                valid_urls.append(u)
        except ValueError:
            continue
    return valid_urls


def resolve_hostname(hostname):
    # Get address information
    addr_info = socket.getaddrinfo(hostname, None)

    # Extract IP addresses from address information
    ipv4_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET]
    ipv6_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET6]

    return ipv4_addresses, ipv6_addresses


class ConcurrentFireCrawlLoader(BaseLoader):
    def __init__(
        self,
        web_path,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        mode: Literal["crawl", "scrape", "map"] = "crawl",
        max_threads: Optional[int] = None,
        params: Optional[Dict] = None,
    ):
        """Concurrent document loader for FireCrawl operations.
        
        Executes multiple FireCrawlLoader instances concurrently using thread pooling
        to improve bulk processing efficiency.

        Args:
            web_path: List of URLs/paths to process.
            api_key: API key for FireCrawl service. Defaults to None 
                (uses FIRE_CRAWL_API_KEY environment variable if not provided).
            api_url: Base URL for FireCrawl API. Defaults to official API endpoint.
            mode: Operation mode selection:
                - 'crawl': Website crawling mode (default)
                - 'scrape': Direct page scraping
                - 'map': Site map generation
            max_threads: Maximum concurrent threads. Defaults to None 
                (auto-configures based on CPU cores).
            params: The parameters to pass to the Firecrawl API.
                Examples include crawlerOptions.
                For more details, visit: https://github.com/mendableai/firecrawl-py
        """
        self.sub_loaders = [
            FireCrawlLoader(
                url=url,
                api_key=api_key,
                api_url=api_url,
                mode=mode,
                params=params
            ) for url in web_path
        ]
        assert max_threads is None or max_threads > 0, "max_threads must be a positive integer."
        if max_threads is not None:
            assert type(max_threads) == int, "max_threads must be None or an integer."
        self.max_threads = max_threads

    def lazy_load(self):
        executor = ThreadPoolExecutor(max_workers=self.max_threads)
        futures = [executor.submit(loader.load) for loader in self.sub_loaders]
        for future in futures:
            try:
                doc = future.result()
                if doc is not None:
                    if type(doc) == list:
                        doc = doc[0]
                    yield doc
            except Exception as e:
                log.error(f"Error processing URL: {e}")


class SafeWebBaseLoader(WebBaseLoader):
    """WebBaseLoader with enhanced error handling for URLs."""

    def __init__(self, trust_env: bool = False, *args, **kwargs):
        """Initialize SafeWebBaseLoader
        Args:
            trust_env (bool, optional): set to True if using proxy to make web requests, for example
                using http(s)_proxy environment variables. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.trust_env = trust_env

    async def _fetch(
        self, url: str, retries: int = 3, cooldown: int = 2, backoff: float = 1.5
    ) -> str:
        async with aiohttp.ClientSession(trust_env=self.trust_env) as session:
            for i in range(retries):
                try:
                    kwargs: Dict = dict(
                        headers=self.session.headers,
                        cookies=self.session.cookies.get_dict(),
                    )
                    if not self.session.verify:
                        kwargs["ssl"] = False

                    async with session.get(
                        url, **(self.requests_kwargs | kwargs)
                    ) as response:
                        if self.raise_for_status:
                            response.raise_for_status()
                        return await response.text()
                except aiohttp.ClientConnectionError as e:
                    if i == retries - 1:
                        raise
                    else:
                        log.warning(
                            f"Error fetching {url} with attempt "
                            f"{i + 1}/{retries}: {e}. Retrying..."
                        )
                        await asyncio.sleep(cooldown * backoff**i)
        raise ValueError("retry count exceeded")

    def _unpack_fetch_results(
        self, results: Any, urls: List[str], parser: Union[str, None] = None
    ) -> List[Any]:
        """Unpack fetch results into BeautifulSoup objects."""
        from bs4 import BeautifulSoup

        final_results = []
        for i, result in enumerate(results):
            url = urls[i]
            if parser is None:
                if url.endswith(".xml"):
                    parser = "xml"
                else:
                    parser = self.default_parser
                self._check_parser(parser)
            final_results.append(BeautifulSoup(result, parser, **self.bs_kwargs))
        return final_results

    async def ascrape_all(
        self, urls: List[str], parser: Union[str, None] = None
    ) -> List[Any]:
        """Async fetch all urls, then return soups for all results."""
        results = await self.fetch_all(urls)
        return self._unpack_fetch_results(results, urls, parser=parser)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path with error handling."""
        for path in self.web_paths:
            try:
                soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
                text = soup.get_text(**self.bs_get_text_kwargs)

                # Build metadata
                metadata = {"source": path}
                if title := soup.find("title"):
                    metadata["title"] = title.get_text()
                if description := soup.find("meta", attrs={"name": "description"}):
                    metadata["description"] = description.get(
                        "content", "No description found."
                    )
                if html := soup.find("html"):
                    metadata["language"] = html.get("lang", "No language found.")

                yield Document(page_content=text, metadata=metadata)
            except Exception as e:
                # Log the error and continue with the next URL
                log.error(f"Error loading {path}: {e}")

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async lazy load text from the url(s) in web_path."""
        results = await self.ascrape_all(self.web_paths)
        for path, soup in zip(self.web_paths, results):
            text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = {"source": path}
            if title := soup.find("title"):
                metadata["title"] = title.get_text()
            if description := soup.find("meta", attrs={"name": "description"}):
                metadata["description"] = description.get(
                    "content", "No description found."
                )
            if html := soup.find("html"):
                metadata["language"] = html.get("lang", "No language found.")
            yield Document(page_content=text, metadata=metadata)

    async def aload(self) -> list[Document]:
        """Load data into Document objects."""
        return [document async for document in self.alazy_load()]


def get_web_loader(
    urls: Union[str, Sequence[str]],
    loader_type: Literal["web", "firecrawl"] = "web",
    verify_ssl: bool = True,
    requests_per_second: int = 2,
    trust_env: bool = False,
    max_threads: Optional[int] = None,
    firecrawl_api_key: Optional[str] = None,
    firecrawl_api_url: Optional[str] = None,
):
    # Check if the URLs are valid
    safe_urls = safe_validate_urls([urls] if isinstance(urls, str) else urls)
    if loader_type == "web":
        return SafeWebBaseLoader(
            web_path=safe_urls,
            verify_ssl=verify_ssl,
            requests_per_second=requests_per_second,
            continue_on_failure=True,
            trust_env=trust_env
        )
    elif loader_type == "firecrawl":
        return ConcurrentFireCrawlLoader(
            safe_urls,
            api_key=firecrawl_api_key,
            api_url=firecrawl_api_url,
            max_threads=max_threads,
        )
    else:
        raise ValueError(ERROR_MESSAGES.INVALID_LOADER_TYPE)
