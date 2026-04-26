import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def export_to_markdown(report_dict):
    """Formats report dictionary into a Markdown string.

    Dynamically renders all sections present in the report dict.
    """
    md = f"# {report_dict.get('Title', 'Research Report')}\n\n"

    # Ordered sections to render
    sections = [
        "Abstract",
        "Key Findings",
        "Key Concepts",
        "Document Excerpts",
        "Sources",
        "Conclusion",
        "Future Scope",
    ]

    for section in sections:
        content = report_dict.get(section)
        if content:
            md += f"## {section}\n{content}\n\n"

    return md


def export_to_pdf(markdown_text):
    """Translates simple markdown text to a PDF byte flow format using reportlab.
    Returns: bytes of the PDF.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    lines = markdown_text.split('\n')
    for line in lines:
        if line.startswith('# '):
            story.append(Paragraph(line.replace('# ', ''), styles['Title']))
        elif line.startswith('## '):
            story.append(Paragraph(line.replace('## ', ''), styles['Heading2']))
        elif line.startswith('> '):
            # Render blockquotes (document excerpts) in italic style
            clean = line.replace('> ', '', 1)
            story.append(Paragraph(f"<i>{clean}</i>", styles['Normal']))
        elif line.startswith('- '):
            # Render bullet points
            clean = line.replace('- ', '• ', 1).replace('**', '')
            story.append(Paragraph(clean, styles['Normal']))
        elif line.strip() == '':
            story.append(Spacer(1, 10))
        else:
            # Strip basic markdown bold for PDF rendering
            clean = line.replace('**', '')
            story.append(Paragraph(clean, styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
