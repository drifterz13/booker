from pathlib import Path

from booker.utils.content_extractor import ContentExtractor
from booker.utils.outline_extractor import OutlineExtractor


def main() -> None:
    sources = [
        Path("docs") / "WorkingEffectivelyWithLegacyCode.pdf",
        Path("docs") / "High-Performance-Browser-Networking.pdf",
    ]
    for src in sources:
        book = OutlineExtractor(src=src).extract()
        print(
            f"{book.source}: {book.page_count} pages, "
            f"{sum(1 for _ in book.walk())} sections, "
            f"maximum level {book.max_level}"
        )

        extractor = ContentExtractor()
        segments = extractor.extract(book)

        for sample in segments[100:105]:
            print(sample.title, sample.path, sample.text)
