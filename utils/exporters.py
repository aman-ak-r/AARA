import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def export_to_markdown(report_dict):
    """Formats report dictionary into a Markdown string."""
    md = f"# {report_dict.get('Title', 'Research Report')}\n\n"
    md += f"## Abstract\n{report_dict.get('Abstract', 'N/A')}\n\n"
    md += f"## Key Findings\n{report_dict.get('Key Findings', 'N/A')}\n\n"
    md += f"## Sources\n{report_dict.get('Sources', 'N/A')}\n\n"
    md += f"## Conclusion\n{report_dict.get('Conclusion', 'N/A')}\n\n"
    md += f"## Future Scope\n{report_dict.get('Future Scope', 'N/A')}\n"
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
        elif line.strip() == '':
            story.append(Spacer(1, 10))
        else:
            story.append(Paragraph(line, styles['Normal']))
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
