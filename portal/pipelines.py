from itemadapter import ItemAdapter


class PortalPipeline:
    """Portal pipeline to process scraped items."""

    def process_item(self, item, spider):
        """Process each item by stripping whitespace from string fields."""
        adapter = ItemAdapter(item)
        self._strip_values(adapter)
        return item

    def _strip_values(self, adapter: ItemAdapter) -> None:
        """Strip whitespace from string fields in the item."""
        for field in adapter.field_names():
            value = adapter.get(field)
            if isinstance(value, str):
                adapter[field] = value.strip()
            elif isinstance(value, list):
                adapter[field] = [v.strip() for v in value if isinstance(v, str)]
