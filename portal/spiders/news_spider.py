import hashlib
from datetime import datetime

import scrapy

from portal.items import DataExtractorItem

FORMAT_DT = "%Y%m%d.%H.%M.%S"
NOW = datetime.now().strftime(FORMAT_DT)


class NewsSpider(scrapy.Spider):
    """Spider to scrape news articles from yogonet.com."""

    name = "news_spider"
    start_urls = ["file:///Users/fernandocopa/Documents/projects/scraper-ayi/response.html"]
    custom_settings = {
        "FEEDS": {
            f"output/data_result_{NOW}.json": {"format": "json", "overwrite": True},
        }
    }

    def parse(self, response):
        """Parse the quotes from the response."""
        return
        content_hash = hashlib.md5(response.body).hexdigest()[:12]
        yield DataExtractorItem(
            {
                "id": content_hash,
                "url": response.url,
                "site_name": "yogonet.com",
                "html_content": response.body.decode(),
                "collected_at": datetime.now().isoformat(),
                "status_code": response.status,
                "content_length": len(response.body),
                "annotations": {
                    "container_selector": ".contenedor_dato_modulo",
                    "title_selector": ".volanta_titulo div.volanta::text",
                    "kicker_selector": ".volanta_titulo > div.fuente_roboto_slab::text",
                    "image_selector": ".imagen > a > img::attr(src)",
                    "link_selector": ".imagen > a::attr(href)",
                },
            }
        )
