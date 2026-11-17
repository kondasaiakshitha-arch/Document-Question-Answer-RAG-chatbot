import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from documents.models import Document
from documents.services import retrieve_relevant_chunks


class Command(BaseCommand):
    help = "Run a lightweight evaluation suite for retrieval and refusal behavior"

    def handle(self, *args, **options):
        questions_path = Path(settings.BASE_DIR) / "tests" / "eval_questions.json"
        dataset = json.loads(questions_path.read_text(encoding="utf-8"))

        retrieval_hits = 0
        correctness = 0
        refusal_accuracy = 0

        for item in dataset:
            document_filter = None
            if item.get("document"):
                try:
                    doc = Document.objects.get(title__icontains=item["document"])
                    document_filter = [doc.id]
                except Document.DoesNotExist:
                    pass

            chunks = retrieve_relevant_chunks(item["question"], document_filter, top_k=5)
            target_found = bool(chunks)
            if target_found:
                retrieval_hits += 1

            answer_text = ""
            if chunks:
                answer_text = chunks[0].text[:200]

            expected = item["expected_answer"].strip().lower()
            actual = answer_text.strip().lower()
            is_correct = expected in actual or actual in expected or expected == actual

            if is_correct:
                correctness += 1

            if item.get("trap"):
                if "couldn't find this in the uploaded documents" in actual.lower() or "i couldn't find" in actual.lower():
                    refusal_accuracy += 1

        total = len(dataset)
        self.stdout.write(f"retrieval_hit_rate={retrieval_hits / total:.2%}")
        self.stdout.write(f"answer_correctness={correctness / total:.2%}")
        self.stdout.write(f"refusal_accuracy={refusal_accuracy / max(1, sum(1 for i in dataset if i.get('trap'))):.2%}")
