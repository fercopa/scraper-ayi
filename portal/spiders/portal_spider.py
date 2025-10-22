import scrapy

from adaptive_scraper.adaptive_extractor import AdaptiveExtractor
from portal.items import PortalItem


class PortalSpider(scrapy.Spider):
    """Spider to scrape news articles from yogonet.com."""

    name = "portal_spider"
    start_urls = [
        "https://www.yogonet.com/international/",
        "https://www.yogonet.com/international/regions/canada/",
    ]
    custom_settings = {
        "FEEDS": {
            "output/final_result.json": {"format": "json", "overwrite": True},
        }
    }

    def __init__(self, with_predictions="false", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_predictions = with_predictions.lower() == "true"

    def parse(self, response):
        """Parse the quotes from the response."""
        if self.with_predictions:
            for article in self.parse_with_prediction(response):
                yield PortalItem(
                    {
                        "title": article.get("title"),
                        "kicker": article.get("kicker"),
                        "image": article.get("image"),
                        "link": article.get("link"),
                    }
                )
        else:
            for element in response.css(".contenedor_dato_modulo"):
                yield PortalItem(
                    {
                        "title": element.css(".volanta_titulo div.volanta::text").get(),
                        "kicker": element.css(".volanta_titulo > div.fuente_roboto_slab::text").get(),
                        "image_url": element.css(".imagen > a > img::attr(src)").get(),
                        "link": element.css(".imagen > a::attr(href)").get(),
                    }
                )

    def parse_with_prediction(self, response):
        """Parse articles using AdaptiveExtractor with ML weights."""
        extractor_ml = AdaptiveExtractor(use_ml_weights=True)
        html = response.text
        return extractor_ml.extract(html, url=response.url)
