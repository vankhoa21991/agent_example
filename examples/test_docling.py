from docling.document_converter import DocumentConverter
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,

)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
import os
import time
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

def example1():
    # accelerator_options = AcceleratorOptions(
    #         num_threads=8, device=AcceleratorDevice.CPU
    #     )

    accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.MPS
        )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    source = "https://arxiv.org/pdf/2403.17834"  # document per local path or URL
    source = "https://medium.com/@baptisteloquette.entr/langchain-arxiv-tutor-long-text-summarization-retrievalqa-and-vector-databases-6d5cb1dc7e14"
    # source = "https://arxiv.org/pdf/2501.14548"
    source = "https://www.sciencedirect.com/science/article/pii/S0959804924018215"
    # source = "https://www.nature.com/articles/s41467-025-56822-w"
    # source = "https://www.sciencedirect.com/science/article/abs/pii/S0031320324010458"
    source = "https://medium.com/@baptisteloquette.entr/langchain-arxiv-tutor-long-text-summarization-retrievalqa-and-vector-databases-6d5cb1dc7e14"
    source = "https://www.nature.com/articles/s41392-024-02111-9"
    source = "https://arxiv.org/pdf/2502.05119"

    source = "assets/Chap1.pdf"
    converter = DocumentConverter(format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
                InputFormat.IMAGE: TesseractOcrOptions(),
                InputFormat.DOCX: None,
                InputFormat.HTML: None,
            })
    result = converter.convert(source)
    # print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"

    print(result)


    # save markdown to file
    with open("output/output.md", "w") as f:
        f.write(result.document.export_to_markdown())

    summarize(result.document.export_to_markdown())

def summarize(text):
    messages = [
        {
            "role": "user",
            "content": f"""Summarize the content in this document: {text}, make it understandable, and concise on important points, make it long enough to cover the
        """
        }
    ]

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
    )

    print(chat_completion.choices[0].message.content)


def example2():
    import urllib.request
    from urllib.request import Request
    from io import BytesIO
    from docling.backend.html_backend import HTMLDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument

    url = "https://medium.com/@yumaueno/a-27-year-old-man-who-grew-his-ai-product-to-over-70k-month-in-just-one-year-from-launch-bedf6732e754"
    url = "https://www.sciencedirect.com/science/article/abs/pii/S0031320324010458"
    url = "https://www.nature.com/articles/s41392-024-02111-9"
    url = "https://arxiv.org/pdf/2502.05119"
    text = urllib.request.urlopen(url).read()
    # text = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    # text = urllib.request.urlopen(text).read()
    in_doc = InputDocument(
        path_or_stream=BytesIO(text),
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="duck.html",
    )
    backend = HTMLDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(text))
    dl_doc = backend.convert()
    # print(dl_doc.name)
    print(dl_doc.export_to_markdown())
    summarize(dl_doc.export_to_markdown())

if __name__=="__main__":
    example1()
    # example2()
