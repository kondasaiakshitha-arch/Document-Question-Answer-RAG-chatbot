from django.conf import settings
from django.db import models

from documents.models import Document


class EvalQuestion(models.Model):
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="eval_questions")
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="eval_questions")
    question = models.TextField()
    expected_answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.question[:80]
