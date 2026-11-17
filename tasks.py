from celery import shared_task

from .models import Document
from .services import index_document


@shared_task
def process_document(document_id):
    document = Document.objects.get(id=document_id)
    document.status = "processing"
    document.save(update_fields=["status"])
    return index_document(document)
