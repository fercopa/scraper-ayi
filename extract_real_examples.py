import json

from bs4 import BeautifulSoup


def extract_examples_from_pattern_2():
    """Extract from Pattern 2 which has distinct title and kicker."""
    with open("output/data_result_20251018.22.24.11.json") as f:
        data = json.load(f)[0]

    html = data["html_content"]
    soup = BeautifulSoup(html, "html.parser")

    containers = soup.select(".contenedor_dato_modulo")

    titles = []
    kickers = []

    for container in containers[:50]:
        # Kicker: .volanta_titulo div.volanta (the SHORT text)
        kicker_elem = container.select_one(".volanta_titulo div.volanta")
        if kicker_elem:
            kicker = kicker_elem.get_text(strip=True)
            if kicker and len(kicker) > 5 and kicker not in kickers:
                kickers.append(kicker)

        # Title: h2.titulo a (the LONG text)
        title_elem = container.select_one("h2.titulo a")
        if title_elem:
            title = title_elem.get_text(strip=True)
            if title and len(title) > 10 and title not in titles and title != kicker:
                titles.append(title)

    return titles, kickers


def extract_examples_from_pattern_1():
    """Extract from Pattern 1."""
    with open("output/extracted_data_20251018.13.14.39.json") as f:
        data = json.load(f)[0]

    html = data["html_content"]
    soup = BeautifulSoup(html, "html.parser")

    containers = soup.select(".item_noticias")

    titles = []
    dates = []

    for container in containers[:50]:
        # Title
        title_elem = container.select_one(".titulo_item_listado_noticias a")
        if title_elem:
            title = title_elem.get_text(strip=True)
            if title and len(title) > 10 and title not in titles:
                titles.append(title)
    return titles, dates


def main():
    """Main function to extract and save examples."""
    print("=" * 70)
    print("EXTRACTING REAL EXAMPLES")
    print("=" * 70)

    # Pattern 2 - has real title and kicker
    titles_p2, kickers_p2 = extract_examples_from_pattern_2()

    print("\nPattern 2 (contenedor_dato_modulo):")
    print(f"  Titles: {len(titles_p2)}")
    print(f"  Kickers: {len(kickers_p2)}")

    # Pattern 1 - only titles
    titles_p1, _ = extract_examples_from_pattern_1()

    print("\nPattern 1 (item_noticias):")
    print(f"  Titles: {len(titles_p1)}")

    # Combine
    all_titles = list(set(titles_p1 + titles_p2))
    all_kickers = list(set(kickers_p2))

    print("\n" + "=" * 70)
    print("TOTAL UNIQUE EXAMPLES")
    print("=" * 70)
    print(f"Titles: {len(all_titles)}")
    print(f"Kickers: {len(all_kickers)}")

    print("\nSample Titles:")
    for title in all_titles[:5]:
        print(f"  - {title}")

    print("\nSample Kickers:")
    for kicker in all_kickers[:10]:
        print(f"  - {kicker}")

    # Save to JSON
    examples = {
        "titles": all_titles,
        "kickers": all_kickers,
    }

    with open("semantic_examples.json", "w") as f:
        json.dump(examples, f, indent=2)

    print("\nSaved to semantic_examples.json")


if __name__ == "__main__":
    main()
