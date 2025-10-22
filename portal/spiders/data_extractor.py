import hashlib
from datetime import datetime

import scrapy

from portal.items import DataExtractorItem
from portal.spiders.constants import FORMAT_DT

NOW = datetime.now().strftime(FORMAT_DT)


class DataExtractorSpider(scrapy.Spider):
    """Spider to extract data from a web."""

    name = "data_extractor"
    allowed_domains = ["yogonet.com"]
    start_urls = ["https://www.yogonet.com/international/regions/canada/"]
    custom_settings = {
        "FEEDS": {
            f"output/extracted_data_{NOW}.json": {"format": "json", "overwrite": False},
        }
    }

    def parse(self, response):
        """Parse the response and yield DataExtractorItem."""
        return
        for page in range(1, 21):
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
                        "container_selector": ".item_noticias",
                        "title_selector": ".titulo_item_listado_noticias a::text",
                        "kicker_selector": ".volanta_item_listado_noticias a::attr(title)",
                        "image_selector": ".imagen_item_listado_noticias img::attr(src)",
                        "link_selector": ".volanta_item_listado_noticias a::attr(href)",
                    },
                }
            )
            next_page = response.css(".contenedor_general_listado_noticias button.boton_paginador.siguiente").get()
            if next_page:
                go_to = f"?buscar=&pagina={page}"
                yield response.follow(go_to, self.parse)
