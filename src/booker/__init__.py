from pathlib import Path

from booker.chunk.chunker import Chunker
from booker.extractor.content import ContentExtractor
from booker.extractor.outline import OutlineExtractor


def main() -> None:
    # source = Path("docs") / "WorkingEffectivelyWithLegacyCode.pdf"
    source = Path("docs") / "High-Performance-Browser-Networking.pdf"
    book = OutlineExtractor(src=source).extract()

    for section in book.walk():
        print(
            section.level,
            section.title,
            section.start_page,
            section.end_page,
        )

    segments = ContentExtractor().extract(book)[100:110]
    chunks = Chunker(chunk_size=2000, chunk_overlap=200).chunk(segments)

    # for segment in segments:
    #     print(
    #         f"Path: {segment.path}, start: {segment.start_page}, end: {segment.end_page}, sample text: {segment.text[:100]}\n"
    #     )
    #     print(
    #         f"Fragment page numbers: {[fragment.page_number for fragment in segment.fragments]}"
    #     )

    for chunk in chunks:
        print(
            f"Chunk path: {chunk.path}, start - end {chunk.start_page} - {chunk.end_page}, wc: {chunk.word_count}, chunk: {chunk.text}"
        )
