# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class PortalItem(scrapy.Item):
    """Portal item model."""

    kicker: str | None = scrapy.Field()
    title: str | None = scrapy.Field()
    image: str | None = scrapy.Field()
    link: str | None = scrapy.Field()


class DataExtractorItem(scrapy.Item):
    """Data extractor item model."""

    id: str = scrapy.Field()
    url: str = scrapy.Field()
    site_name: str = scrapy.Field()
    html_content: str = scrapy.Field()
    collected_at: str = scrapy.Field()
    status_code: int = scrapy.Field()
    content_length: int = scrapy.Field()
    annotations: dict = scrapy.Field()
